-- //////////////////////////////////////////
-- Typical flow:
-- require 'unimodal'
-- loadPackagesAndModel(datasetType, modality) -- 'mir' or 'nus', 'I', or 'X'
-- loadData() -- uses dataLoader.lua
-- optional: loadModelSnapshot -- from createModel.lua
-- trainAndEvaluate(numEpochs, batchSize)
-- /////////////////////////////////////////

function doSetLRForLayer(layerIdx, newLRMult)
    m.classifier:get(layerIdx):learningRate('weight', newLRMult)
    m.classifier:get(layerIdx):learningRate('bias', newLRMult)
end

function doSetWDForLayer(layerIdx, newWDMult)
    m.classifier:get(layerIdx):weightDecay('weight', newWDMult)
    m.classifier:get(layerIdx):weightDecay('bias', newWDMult)
end

local function doChangeLRAndWDForLayer(layerName, newLR, newWD)

    local newLRMult = newLR / p.baseLearningRate
    local newWDMult
    if p.baseWeightDecay == 0 then
        newWDMult = 1
    else
        newWDMult = newWD / p.baseWeightDecay
    end

    if layerName == 'feature' then
        print('Changing feature learning rate not yet implemented')
    elseif layerName == 'top' then
        p.topLearningRate = newLR
        doSetLRForLayer(17, newLRMult)
        doSetLRForLayer(20, newLRMult)
        doSetWDForLayer(17, newWDMult)
        doSetWDForLayer(20, newWDMult)
    elseif layerName == 'last' then
        p.lastLayerLearningRate = newLR
        doSetLRForLayer(23, newLRMult)
        doSetWDForLayer(23, newWDMult)
    end

    -- This still needs a call to setOptimStateLRAndWD to be complete
end

function setOptimStateLRAndWD(newLR, newWD)
    local learningRates, weightDecays = m.classifier:getOptimConfig(newLR or p.baseLearningRate, newWD or p.baseWeightDecay)
    g.optimState.learningRates = learningRates
    g.optimState.weightDecays = weightDecays
end

function changeLRAndWDForLayer(layerName, newLR, newWD)
    doChangeLRAndWDForLayer(layerName, newLR, newWD)
    setOptimStateLRAndWD(p.baseLearningRate, p.baseWeightDecay)
end

function loadPackagesAndModel(datasetType, modality, baseLearningRate, baseWeightDecay, mom)

    require 'nn'
    require 'optim'
    nninit = require 'nninit'
    gnuplot = require 'gnuplot'
    require 'auxf.evaluate'
    require 'auxf.dataLoader'
    require 'auxf.createModel'
    require 'imagenetloader.dataset'
    require 'cutorch'
    require 'cunn'
    require 'cudnn'

    require 'inn'
    require 'nnlr'
    loadcaffe = require 'loadcaffe'

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
        snapshotDatasetDir = '/mirflickr'
        g.datasetPath = '/home/kjoslyn/datasets/mirflickr/'
    elseif datasetType == 'nus' then
        p.numClasses = 21
        p.tagDim = 1000
        snapshotDatasetDir = '/nuswide'
        g.datasetPath = '/home/kjoslyn/datasets/nuswide/'
    else
        print("Error: Unrecognized datasetType!! Should be mir or nus")
    end
    p.datasetType = datasetType
    p.modality = modality
    p.batchSize = 100

    g.snapshotDir = '/home/kjoslyn/kevin/Project/snapshots' .. snapshotDatasetDir

    g.numEpochsCompleted = 0

    if datasetType == 'nus' and modality == 'I' then
        g.plotNumEpochs = 1
    else
        g.plotNumEpochs = 5
    end

    -- //////////// Load image / text model
    if modality == 'I' then
    --    m.classifier = getImageModel()
       m.classifier = getImageModelImageNetPretrained(1e3)
       m.imageClassifier = m.classifier
       if datasetType == 'nus' then
          g.evalTrainAccEpochs = 1
       else
          g.evalTrainAccEpochs = 5
       end
    elseif modality == 'X' then
       m.classifier = getUntrainedTextModel()
       m.textClassifier = m.classifier
       g.evalTrainAccEpochs = 10
    else
       print('Error in unimodal.lua: Unrecognized modality')
    end

    g.accIdx = 0
    g.pastNAcc = torch.Tensor(g.plotNumEpochs)
    g.avgDataAcc = torch.Tensor()
    g.maxDataAcc = torch.Tensor()
    g.minDataAcc = torch.Tensor()
    g.pastNLoss = torch.Tensor(g.plotNumEpochs)
    g.avgDataLoss = torch.Tensor()
    g.maxDataLoss = torch.Tensor()
    g.minDataLoss = torch.Tensor()

    -- TODO: Fix image-text disparity
    if modality == 'I' then
        p.baseLearningRate = baseLearningRate or 1e-4
    else
        p.baseLearningRate = baseLearningRate or .1
    end
    p.topLearningRate = 0.01
    p.lastLayerLearningRate = 0.1

    p.baseWeightDecay = baseWeightDecay or 0 -- 1e-4
    p.topWeightDecay =  1 -- .01
    p.lastLayerWeightDecay =  1 -- .02

    p.baseLearningRateDecay = 0
    p.baseMomentum = mom or 0.9

    g.optimState = {
        learningRate = p.baseLearningRate,
        learningRateDecay = p.baseLearningRateDecay,
        momentum = p.baseMomentum
    }

    if p.modality == 'I' then
        doChangeLRAndWDForLayer('top', p.topLearningRate, p.topWeightDecay)
        doChangeLRAndWDForLayer('last', p.lastLayerLearningRate, p.lastLayerWeightDecay)
    end
    setOptimStateLRAndWD(p.baseLearningRate, p.baseWeightDecay)
end

function doGetBatchFromMemory(startIndex, endIndex, perm)

    local batchPerm = perm[ {{ startIndex, endIndex }} ]
    local batch = {}
    batch.data = d.trainset.data:index(1, batchPerm)
    batch.label = d.trainset.label:index(1, batchPerm)

    return batch
end

function doGetBatchImageFromDataset(startIndex, endIndex, perm)

    local batch = {}
    batch.data, batch.label = d.dataset:getBySplit({'training', 'pretraining'}, 'I', startIndex, endIndex, perm)
    -- batch.data, batch.label = d.dataset:getBySplit({'training'}, 'I', startIndex, endIndex, perm)

    return batch
end

function getBatch(batchNum, batchSize, perm)

    local startIndex = batchNum * batchSize + 1
    local endIndex = math.min((batchNum + 1) * batchSize, Ntrain)

    local batch
    if p.modality == 'I' and p.datasetType == 'nus' then
        batch = doGetBatchImageFromDataset(startIndex, endIndex, perm)
    else
        batch = doGetBatchFromMemory(startIndex, endIndex, perm)
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

    local imageRootPath = g.datasetPath .. 'ImageData'
    d.dataset = imageLoader{path=imageRootPath, sampleSize={3,227,227}, splitFolders={'training', 'pretraining', 'val', 'query'}}

    Ntrain = d.dataset:sizeTrain() + d.dataset:sizePretraining()
    -- Ntrain = d.dataset:sizeTrain() + d.dataset:sizePretraining() + d.dataset:sizeVal()
    -- Ntrain = d.dataset:sizeTrain()
    Ntest = d.dataset:sizeTest()
    Nval = d.dataset:sizeVal()

    d.trainset = {}
    d.valset = {}
    d.testset = {}
    if p.modality == 'X' then
        -- This will actually be used in training, and in evaluating the trainset accuracy
        if not small then
            d.trainset.data, d.trainset.label = d.dataset:getBySplit({'training', 'pretraining'}, 'X', 1, Ntrain)
            -- d.trainset.data, d.trainset.label = d.dataset:getBySplit({'training', 'pretraining', 'val'}, 'X', 1, Ntrain)
        else
            Ntrain = d.dataset:sizeTrain()
            d.trainset.data, d.trainset.label = d.dataset:getBySplit({'training'}, 'X', 1, Ntrain)
        end
    else
        if p.datasetType == 'nus' then
            -- This will only be used in evaluating the trainset accuracy. Training will include pretraining
            -- The entire trainset (including pretrain) is too larget to hold in memory
            d.trainset.data, d.trainset.label = d.dataset:getBySplit('training', 'I', 1, d.dataset:sizeTrain())
        else
            d.trainset.data, d.trainset.label = d.dataset:getBySplit({'training', 'pretraining'}, 'I', 1, Ntrain)
        end
    end
    d.valset.data, d.valset.label = d.dataset:getBySplit('val', p.modality, 1, Nval)
    d.testset.data, d.testset.label = d.dataset:getBySplit('query', p.modality, 1, Ntest)

    collectgarbage()
end

function trainAndEvaluate(numEpochs, batchSize, lr, mom, wd)

    local batchSize = batchSize or p.batchSize

    -- lr and wd parameters should only be used when not setting different learning rates for different layers
    -- (i.e. text modality)
    if lr or wd then
        local lr = lr or p.baseLearningRate
        local wd = wd or p.baseWeightDecay
        setOptimStateLRAndWD(lr, wd)
    end
    if mom then
        g.optimState.momentum = mom
    end

    local startEpoch = g.numEpochsCompleted + 1

    if not g.plotStartEpoch then
        g.plotStartEpoch = startEpoch
    end

    -- criterion = nn.MultiLabelSoftMarginCriterion()
    -- criterion = nn.MSECriterion()
    criterion = nn.BCECriterion()
    if p.modality == 'I' or p.noSizeAverage then
        criterion.sizeAverage = false -- TODO: Why does this work better for image modality?
    end
    criterion = criterion:cuda()

    local params, gradParams = m.classifier:getParameters()

    -- g.optimState.learningRate = learningRate

    -- local optimState = {
    --     learningRate = learningRate, -- .01 works for mirflickr
    --     -- learningRateDecay = 1e-7
    --     -- learningRate = 1e-3,
    --     -- learningRateDecay = 1e-4,
    --     weightDecay = weightDecay, -- .01?
    --     momentum = momentum -- 0.9?
    -- }

    numBatches = math.ceil(Ntrain / batchSize)

    m.classifier:training()

    local epochTimer = torch.Timer()

    for epoch = startEpoch, startEpoch + numEpochs - 1 do

        epochTimer:reset()
        epochTimer:resume()
        -- shuffle at each epoch
        perm = torch.randperm(Ntrain):long()
        totalLoss = 0
        totNumIncorrect = 0

        for batchNum = 0, numBatches - 1 do

            trainBatch = getBatch(batchNum, batchSize, perm)
            collectgarbage()

            function feval(x)
                -- get new parameters
                if x ~= params then
                    params:copy(x)
                end

                gradParams:zero()

                outputs = m.classifier:forward(trainBatch.data)
                local loss = criterion:forward(outputs, trainBatch.label)

                local roundedOutput = torch.CudaTensor(outputs:size(1), p.numClasses):copy(outputs):round()
                totNumIncorrect = totNumIncorrect + torch.ne(roundedOutput, trainBatch.label):sum()
                -- local numOnes = roundedOutput:sum()

                local dloss_doutputs = criterion:backward(outputs, trainBatch.label)
                m.classifier:backward(trainBatch.data, dloss_doutputs)

                if p.modality == 'I' or p.noSizeAverage then
                    local inputSize = trainBatch.data:size(1)
                    gradParams:div(inputSize)
                    loss = loss/inputSize
                end
                totalLoss = totalLoss + loss

                return loss, gradParams
            end
            optim.sgd(feval, params, g.optimState)

            -- collectgarbage()
        end
        epochTimer:stop()

        collectgarbage()
        m.classifier:evaluate()
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
        print(string.format('Epoch time: %.2f seconds\n', epochTimer:time().real))
        m.classifier:training()

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

        local save
        if p.modality == 'I' then
            save = epoch % 10 == 0
        else
            save = epoch == 500 or epoch == 750 or epoch == 900 or epoch == 1000
        end

        if save then
        -- if epoch == 300 or epoch == 500 or epoch == 750 or epoch == 900 or epoch == 1000 then
            local date = os.date("*t", os.time())
            local dateStr = date.month .. "_" .. date.day .. "_" .. date.hour .. "_" .. date.min
            saveSnapshot(dateStr .. '_snapshot_epoch_' .. epoch)
        end

        g.numEpochsCompleted = g.numEpochsCompleted + 1
    end
end

function saveSnapshot(filename)
    local modalityDir
    if p.modality == 'I' then
        modalityDir = 'imageNet'
    else
        modalityDir = 'textNet'
    end
    local snapshot = {}
    snapshot.params, snapshot.gparams = m.classifier:getParameters()
    snapshot.g = g
    torch.save(g.snapshotDir .. '/' .. modalityDir .. '/' .. filename .. '.t7', snapshot)
end