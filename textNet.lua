
-- //////////////////////////////////////////
-- Typical flow:
-- require 'textNet'
-- loadPackagesAndModel(datasetType) -- 'mir' or 'nus'
-- loadData(useKFold, small) -- uses dataLoader.lua. 1 input parameter- true for 1000 instanes (small)
-- optional: loadModelSnapshot -- from createModel.lua
-- trainAndEvaluate(kFoldNum, batchSize, learningRate, numEpochs, startEpoch, printTestsetAcc)

-- For nuswide, batchsize of 100 and lr of .1 works well. Epoch 500-750 good, overfitting after that
-- /////////////////////////////////////////

function loadPackagesAndModel(datasetType)

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
    require 'imagenetloader.dataset'

    p = {} -- parameters
    d = {} -- data
    m = {} -- models
    g = {} -- other global variables
    o = {} -- optimStates and model parameters

    matio = require 'matio'

    local snapshotDatasetDir
    if datasetType == 'mir' then
        p.numClasses = 24
        p.tagDim = 1075
        g.datasetPath = '/home/kjoslyn/datasets/mirflickr/'
        snapshotDatasetDir = '/mirflickr'
    elseif datasetType == 'nus' then
        p.numClasses = 21
        p.tagDim = 1000
        g.datasetPath = '/home/kjoslyn/datasets/nuswide/'
        snapshotDatasetDir = '/nuswide'
    else
        print("Error: Unrecognized datasetType!! Should be mir or nus")
    end
    p.datasetType = datasetType

    g.snapshotDir = '/home/kjoslyn/kevin/Project/snapshots' .. snapshotDatasetDir

    g.numEpochsCompleted = 0

    -- //////////// Load text model
    
    m.textClassifier = getUntrainedTextModel()

    g.accIdx = 0
    g.plotNumEpochs = 5
    g.evalTrainAccEpochs = 10
    g.pastNAcc = torch.Tensor(g.plotNumEpochs)
    g.avgDataAcc = torch.Tensor()
    g.maxDataAcc = torch.Tensor()
    g.minDataAcc = torch.Tensor()
    g.pastNLoss = torch.Tensor(g.plotNumEpochs)
    g.avgDataLoss = torch.Tensor()
    g.maxDataLoss = torch.Tensor()
    g.minDataLoss = torch.Tensor()
end

function getBatch(batchNum, batchSize, perm)

    startIndex = batchNum * batchSize + 1
    endIndex = math.min((batchNum + 1) * batchSize, Ntrain)

    batchPerm = perm[ {{ startIndex, endIndex }} ]
    batch = {}
    batch.data = d.trainset.data:index(1, batchPerm)
    batch.label = d.trainset.label:index(1, batchPerm)
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

    local imageRootPath = g.datasetPath .. 'ImageData'
    d.dataset = imageLoader{path=imageRootPath, sampleSize={3,227,227}, splitFolders={'training', 'pretraining', 'val', 'query'}}

    Ntrain = d.dataset:sizeTrain() + d.dataset:sizePretraining()
    Ntest = d.dataset:sizeTest()
    Nval = d.dataset:sizeVal()

    d.trainset = {}
    d.valset = {}
    d.testset = {}
    d.trainset.data, d.trainset.label = d.dataset:getBySplit({'training', 'pretraining'}, 'X', 1, Ntrain)
    d.valset.data, d.valset.label = d.dataset:getBySplit('val', 'X', 1, Nval)
    d.testset.data, d.testset.label = d.dataset:getBySplit('query', 'X', 1, Ntest)

    collectgarbage()
end

function trainAndEvaluate(batchSize, learningRate, momentum, weightDecay, numEpochs, printTestsetAcc)

    -- batchSize is 128 for image modality, -1 for text (no batch for text)

    local startEpoch = g.numEpochsCompleted + 1

    if not g.plotStartEpoch then
        g.plotStartEpoch = startEpoch
    end

    -- criterion = nn.MultiLabelSoftMarginCriterion()
    -- criterion = nn.MSECriterion()
    criterion = nn.BCECriterion()
    criterion.sizeAverage = false -- TODO: This is not in image network!

    criterion = criterion:cuda()
    m.textClassifier:cuda()

    params, gradParams = m.textClassifier:getParameters()

    local optimState = {
        learningRate = learningRate, -- .01 works for mirflickr
        -- learningRateDecay = 1e-4,
        weightDecay = weightDecay, -- 0.01
        momentum = momentum -- 0.9
    }

    if batchSize == -1 then
        batchSize = Ntrain
    end

    numBatches = math.ceil(Ntrain / batchSize)

    m.textClassifier:training()

    for epoch = startEpoch, startEpoch + numEpochs - 1 do

        print('gc = ' .. collectgarbage('count'))
        collectgarbage()

        -- shuffle at each epoch
        perm = torch.randperm(Ntrain):long()
        totalLoss = 0
        totNumIncorrect = 0

        for batchNum = 0, numBatches - 1 do

            trainBatch = getBatch(batchNum, batchSize, perm)

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

                --TODO: This is not in image network!!!
                local inputSize = trainBatch.data:size(1)
                gradParams:div(inputSize)
                loss = loss/inputSize

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
        local valClassAcc
        if d.valset then
            valClassAcc = getClassAccuracy(d.valset.data, d.valset.label:cuda())
            print(string.format("Epoch %d: Val set class accuracy = %.2f", epoch, valClassAcc))
        end
        if d.trainset and epoch % g.evalTrainAccEpochs == 0 then
            local trainClassAcc = getClassAccuracy(d.trainset.data, d.trainset.label:cuda())
            print(string.format("Epoch %d: Train set class accuracy = %.2f", epoch, trainClassAcc))
        end
        m.textClassifier:training()

        g.accIdx = g.accIdx + 1
        g.pastNAcc[g.accIdx] = valClassAcc
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
            local date = os.date("*t", os.time())
            local dateStr = date.month .. "_" .. date.day .. "_" .. date.hour .. "_" .. date.min
            local snapshotFile = g.snapshotDir .. "/textNet/" .. dateStr .. "_snapshot_epoch_" .. epoch .. ".t7" 
            local snapshot = {}
            snapshot.params = paramsToSave
            snapshot.gparams = gp
            torch.save(snapshotFile, snapshot)
        end

        g.numEpochsCompleted = g.numEpochsCompleted + 1
    end
end