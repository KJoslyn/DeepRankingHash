
-- //////////////////////////////////////////
-- Typical flow:
-- require 'imageNet'
-- loadData() -- uses dataLoader.lua. 1 input parameter- true for 1000 instanes (small)
-- optional: loadModelSnapshot -- from createModel.lua
-- trainAndEvaluate()
-- /////////////////////////////////////////

require 'nn'
require 'loadcaffe' -- doesn't work on server
require 'image'
require 'optim'
nninit = require 'nninit'
require 'auxf.evaluate'
require 'auxf.dataLoader'
require 'auxf.createModel'

GPU = true
if (GPU) then
    require 'cutorch'
    require 'cunn'
    require 'cudnn'
end

matio = require 'matio'

dataPath = '/home/kjoslyn/torch/test/data/mirflickr/' -- server
-- dataPath = '../../Datasets/mirflickr/' -- labcomp
filePath = '/home/kjoslyn/kevin/' -- server
snapshotDir = '/home/kjoslyn/kevin/Project/snapshots'

-- //////////// Load image model
imageClassifier = getImageModel()

function getBatch(kFoldNum, batchNum, batchSize, perm)

    startIndex = batchNum * batchSize + 1
    endIndex = math.min((batchNum + 1) * batchSize, Ntrain)

    batchPerm = perm[ {{ startIndex, endIndex }} ]
    batch = {}
    if kFoldNum == -1 then
        batch.data = trainset.data:index(1, batchPerm)
        batch.label = trainset.label:index(1, batchPerm)
    else
        batch.data = KFoldTrainSet[kFoldNum].data:index(1, batchPerm)
        batch.label = KFoldTrainSet[kFoldNum].label:index(1, batchPerm)
    end
    batch.data = batch.data:cuda()
    batch.label = batch.label:cuda()

    setmetatable(batch, 
        {__index = function(t, i) 
                        return {t.data[i], t.label[i]} 
                    end}
    );

    function batch:size() 
        return self.data:size(1) 
    end

    return batch
end

function loadData(small)

    trainset, testset = getImageData(small)
    totTrain = trainset.data:size(1)
    Ntest = testset.data:size(1)
    KFoldTrainSet, KFoldValSet = getKFoldSplit(trainset, 5)
end

function getKFoldSplit(trainset, K)

    local sizeVal = math.ceil(totTrain / K)

    KFoldTrainSet = {}
    KFoldValSet = {}

    for k = 1,K do
        valStartIdx = (k-1) * sizeVal + 1
        valEndIdx = math.min(k * sizeVal, totTrain)

        KFoldTrainSet[k] = {}
        KFoldValSet[k] = {}

        KFoldValSet[k].data = trainset.data[ {{ valStartIdx, valEndIdx }} ]
        KFoldValSet[k].label = trainset.label[ {{ valStartIdx, valEndIdx }} ]

        sizeTrain = totTrain - KFoldValSet[k].data:size(1)
        KFoldTrainSet[k].data = torch.FloatTensor(sizeTrain, 3, 227, 227)
        KFoldTrainSet[k].label = torch.LongTensor(sizeTrain, 24)

        local markerIdx = 1
        if valStartIdx ~= 1 then
            KFoldTrainSet[k].data[ {{ 1, valStartIdx - 1 }} ] = trainset.data[ {{ 1, valStartIdx - 1 }} ]
            KFoldTrainSet[k].label[ {{ 1, valStartIdx - 1 }} ] = trainset.label[ {{ 1, valStartIdx - 1 }} ]
            markerIdx = valStartIdx
        end 
        if valEndIdx ~= totTrain then
            KFoldTrainSet[k].data[ {{ markerIdx, sizeTrain }} ] = trainset.data[ {{ valEndIdx + 1, totTrain }} ]
            KFoldTrainSet[k].label[ {{ markerIdx, sizeTrain }} ] = trainset.label[ {{ valEndIdx + 1, totTrain }} ]
        end 
    end

    return KFoldTrainSet, KFoldValSet
end

function trainAndEvaluate(kFoldNum, numEpochs, startEpoch)
    -- kFoldNum is the number of the validation set that will be used for training in this run
    -- use -1 for no validation set

    -- criterion = nn.MultiLabelSoftMarginCriterion()
    -- criterion = nn.MSECriterion()
    criterion = nn.BCECriterion()

    if (GPU) then
        criterion = criterion:cuda()
        testset.data = testset.data:cuda()
        testset.label = testset.label:cuda()
        imageClassifier:cuda()
    end

    params, gradParams = imageClassifier:getParameters()

    local optimState = {
        learningRate = .01
        -- learningRateDecay = 1e-7
        -- learningRate = 1e-3,
        -- learningRateDecay = 1e-4,
        -- weightDecay = 0.01
        -- momentum = 0.9
    }

    batchSize = 128

    if kFoldNum == -1 then
        Nval = 0
    else
        Nval = KFoldValSet[kFoldNum].data:size(1)
    end
    Ntrain = totTrain - Nval 

    numBatches = math.ceil(Ntrain / batchSize)

    imageClassifier:training()

    if not startEpoch then
        startEpoch = 1
    end
    for epoch = startEpoch, numEpochs do

        -- shuffle at each epoch
        perm = torch.randperm(Ntrain):long()
        totalLoss = 0
        totNumIncorrect = 0

        for batchNum = 0, numBatches - 1 do

            trainBatch = getBatch(kFoldNum, batchNum, batchSize, perm)

            function feval(x)
                -- get new parameters
                if x ~= params then
                    params:copy(x)
                end

                gradParams:zero()

                local outputs = imageClassifier:forward(trainBatch.data)
                local loss = criterion:forward(outputs, trainBatch.label)

                totalLoss = totalLoss + loss
                local roundedOutput = torch.CudaTensor(outputs:size(1), 24):copy(outputs):round()
                totNumIncorrect = totNumIncorrect + torch.ne(roundedOutput, trainBatch.label):sum()
                -- local numOnes = roundedOutput:sum()

                local dloss_doutputs = criterion:backward(outputs, trainBatch.label)
                imageClassifier:backward(trainBatch.data, dloss_doutputs)

                return loss, gradParams
            end
            optim.sgd(feval, params, optimState)

            -- collectgarbage()
        end

        imageClassifier:evaluate()
        avgLoss = totalLoss / numBatches
        print("Epoch " .. epoch .. ": avg loss = " .. avgLoss)
        avgNumIncorrect = totNumIncorrect / Ntrain
        print("Epoch " .. epoch .. ": avg num incorrect = " .. avgNumIncorrect)
        if kFoldNum ~= -1 then
            classAcc = calcClassAccuracy(KFoldValSet[kFoldNum].data, KFoldValSet[kFoldNum].label:cuda())
            print(string.format("Epoch %d: Val set class accuracy = %.2f\n", epoch, classAcc))
        end
        imageClassifier:training()

        if epoch == 500 or epoch == 750 or epoch == 900 or epoch == 1000 then
            local paramsToSave, gp = imageClassifier:getParameters()
            local snapshotFile = snapshotDir .. "/imageNet/snapshot_epoch_" .. epoch .. ".t7" 
            local snapshot = {}
            snapshot.params = paramsToSave
            snapshot.gparams = gp
            torch.save(snapshotFile, snapshot)
        end
    end
end