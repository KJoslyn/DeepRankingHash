
-- //////////////////////////////////////////
-- Typical flow:
-- require 'textNet'
-- loadPackagesAndModel(dataset) -- 'mir' or 'nus'
-- loadData(useKFold, small) -- uses dataLoader.lua. 1 input parameter- true for 1000 instanes (small)
-- optional: loadModelSnapshot -- from createModel.lua
-- trainAndEvaluate(kFoldNum, batchSize, learningRate, numEpochs, startEpoch, printTestsetAcc)

-- For nuswide, batchsize of 100 and lr of .1 works well. Epoch 500-750 good, overfitting after that
-- /////////////////////////////////////////

function loadPackagesAndModel(dataset)

    require 'nn'
    require 'optim'
    nninit = require 'nninit'
    gnuplot = require 'gnuplot'
    require 'auxf.evaluate'
    require 'auxf.dataLoader'
    require 'auxf.createModel'
    require 'cutorch'
    require 'cunn'
    require 'cudnn'

    p = {} -- parameters
    d = {} -- data
    m = {} -- models
    g = {} -- other global variables
    o = {} -- optimStates and model parameters

    matio = require 'matio'

    local snapshotDatasetDir
    if dataset == 'mir' then
        p.numClasses = 24
        snapshotDatasetDir = '/mirflickr'
    elseif dataset == 'nus' then
        p.numClasses = 21
        snapshotDatasetDir = '/nuswide'
    else
        print("Error: Unrecognized dataset!! Should be mir or nus")
    end
    p.datasetType = dataset

    g.filePath = '/home/kjoslyn/kevin/' -- server
    g.snapshotDir = '/home/kjoslyn/kevin/Project/snapshots' .. snapshotDatasetDir

    -- //////////// Load text model
    m.textClassifier = getTextModelForNuswide()

    g.accIdx = 0
    g.plotNumEpochs = 5;
    g.pastNAcc = torch.Tensor(g.plotNumEpochs)
    g.avgDataAcc = torch.Tensor()
    g.maxDataAcc = torch.Tensor()
    g.minDataAcc = torch.Tensor()
    g.pastNLoss = torch.Tensor(g.plotNumEpochs)
    g.avgDataLoss = torch.Tensor()
    g.maxDataLoss = torch.Tensor()
    g.minDataLoss = torch.Tensor()
end

function getBatch(kFoldNum, batchNum, batchSize, perm)

    startIndex = batchNum * batchSize + 1
    endIndex = math.min((batchNum + 1) * batchSize, Ntrain)

    batchPerm = perm[ {{ startIndex, endIndex }} ]
    batch = {}
    if kFoldNum == -1 then
        batch.data = d.trainset.data:index(1, batchPerm)
        batch.label = d.trainset.label:index(1, batchPerm)
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

function loadData(useKFold, small)

    if p.datasetType == 'mir' then
        d.trainset, d.testset = getTextDataMirflickr(small)
    elseif p.datasetType == 'nus' then
        d.trainset, d.testset, d.valset = getTextDataNuswide(small)
    end

    totTrain = d.trainset.data:size(1)
    Ntest = d.testset.data:size(1)

    if useKFold then
        KFoldTrainSet, KFoldValSet = getKFoldSplit(d.trainset, 5)
    end
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

        KFoldValSet[k].data = d.trainset.data[ {{ valStartIdx, valEndIdx }} ]
        KFoldValSet[k].label = d.trainset.label[ {{ valStartIdx, valEndIdx }} ]

        sizeTrain = totTrain - KFoldValSet[k].data:size(1)
        KFoldTrainSet[k].data = torch.FloatTensor(sizeTrain, 3, 227, 227)
        KFoldTrainSet[k].label = torch.LongTensor(sizeTrain, p.numClasses)

        markerIdx = 1
        if valStartIdx ~= 1 then
            KFoldTrainSet[k].data[ {{ 1, valStartIdx - 1 }} ] = d.trainset.data[ {{ 1, valStartIdx - 1 }} ]
            KFoldTrainSet[k].label[ {{ 1, valStartIdx - 1 }} ] = d.trainset.label[ {{ 1, valStartIdx - 1 }} ]
            markerIdx = valStartIdx
        end 
        if valEndIdx ~= totTrain then
            KFoldTrainSet[k].data[ {{ markerIdx, sizeTrain }} ] = d.trainset.data[ {{ valEndIdx + 1, totTrain }} ]
            KFoldTrainSet[k].label[ {{ markerIdx, sizeTrain }} ] = d.trainset.label[ {{ valEndIdx + 1, totTrain }} ]
        end 
    end

    return KFoldTrainSet, KFoldValSet
end

function trainAndEvaluate(kFoldNum, batchSize, learningRate, numEpochs, startEpoch, printTestsetAcc)
    -- kFoldNum is the number of the validation set that will be used for training in this run
    -- use -1 for no validation set

    -- batchSize is 128 for image modality, -1 for text (no batch for text)

    if not g.plotStartEpoch then
        g.plotStartEpoch = startEpoch
    end

    -- criterion = nn.MultiLabelSoftMarginCriterion()
    -- criterion = nn.MSECriterion()
    criterion = nn.BCECriterion()
    -- criterion.sizeAverage = false -- TODO: This is not in image network!

    criterion = criterion:cuda()
    -- d.testset.data = d.testset.data:cuda()
    -- d.testset.label = d.testset.label:cuda()
    m.textClassifier:cuda()

    params, gradParams = m.textClassifier:getParameters()

    local optimState = {
        learningRate = learningRate -- .01 works for mirflickr
        -- learningRateDecay = 1e-7
        -- learningRate = 1e-3,
        -- learningRateDecay = 1e-4,
        -- weightDecay = 0.01
        -- momentum = 0.9
    }

    if batchSize == -1 then
        batchSize = Ntrain
    end

    if kFoldNum == -1 then
        Nval = 0
    else
        Nval = KFoldValSet[kFoldNum].data:size(1)
    end
    Ntrain = totTrain - Nval 

    numBatches = math.ceil(Ntrain / batchSize)

    m.textClassifier:training()

    if not startEpoch then
        startEpoch = 1
    end
    for epoch = startEpoch, numEpochs do

        print('gc = ' .. collectgarbage('count'))
        collectgarbage()

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

                local outputs = m.textClassifier:forward(trainBatch.data)
                local loss = criterion:forward(outputs, trainBatch.label)

                totalLoss = totalLoss + loss
                local roundedOutput = torch.CudaTensor(outputs:size(1), p.numClasses):copy(outputs):round()
                totNumIncorrect = totNumIncorrect + torch.ne(roundedOutput, trainBatch.label):sum()
                -- local numOnes = roundedOutput:sum()

                local dloss_doutputs = criterion:backward(outputs, trainBatch.label)
                m.textClassifier:backward(trainBatch.data, dloss_doutputs)

                -- TODO: This is not in image network!!!
                -- local inputSize = trainBatch.data:size(1)
                -- gradParams:div(inputSize)
                -- loss = loss/inputSize

                return loss, gradParams
            end
            optim.sgd(feval, params, optimState)

            -- collectgarbage()
        end

        m.textClassifier:evaluate()
        avgLoss = totalLoss / numBatches
        print("Epoch " .. epoch .. ": avg loss = " .. avgLoss)
        avgNumIncorrect = totNumIncorrect / Ntrain
        print("Epoch " .. epoch .. ": avg num incorrect = " .. avgNumIncorrect)
        if kFoldNum ~= -1 then
            classAcc = getClassAccuracy(KFoldValSet[kFoldNum].data, KFoldValSet[kFoldNum].label:cuda())
            print(string.format("Epoch %d: Val set class accuracy = %.2f\n", epoch, classAcc))
        elseif d.valset then
            classAcc = getClassAccuracy(d.valset.data, d.valset.label:cuda())
            print(string.format("Epoch %d: Val set class accuracy = %.2f\n", epoch, classAcc))
        elseif printTestsetAcc then
            classAcc = getClassAccuracy(d.testset.data, d.testset.label:cuda())
            print(string.format("Epoch %d: Test set class accuracy = %.2f\n", epoch, classAcc))
        end
        m.textClassifier:training()

        g.accIdx = g.accIdx + 1
        g.pastNAcc[g.accIdx] = classAcc
        g.pastNLoss[g.accIdx] = math.min(avgLoss * 100, 100) -- Scale loss to fit in the same y axes as accuracy
        if g.accIdx % g.plotNumEpochs == 0 then
            g.avgDataAcc = g.avgDataAcc:cat(torch.Tensor({g.pastNAcc:mean()}))
            g.maxDataAcc = g.maxDataAcc:cat(torch.Tensor({g.pastNAcc:max()}))
            g.minDataAcc = g.minDataAcc:cat(torch.Tensor({g.pastNAcc:min()}))
            g.avgDataLoss = g.avgDataLoss:cat(torch.Tensor({g.pastNLoss:mean()}))
            g.maxDataLoss = g.maxDataLoss:cat(torch.Tensor({g.pastNLoss:max()}))
            g.minDataLoss = g.minDataLoss:cat(torch.Tensor({g.pastNLoss:min()}))
            plotClassAccAndLoss(epoch, true)
            g.accIdx = 0
        end

        if epoch == 500 or epoch == 750 or epoch == 900 or epoch == 1000 then
            local paramsToSave, gp = m.textClassifier:getParameters()
            local snapshotFile = g.snapshotDir .. "/textNet/snapshot_epoch_" .. epoch .. ".t7" 
            local snapshot = {}
            snapshot.params = paramsToSave
            snapshot.gparams = gp
            torch.save(snapshotFile, snapshot)
        end
    end
end