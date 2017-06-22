-- //////////////////////////////////////////
-- Typical flow:
-- require 'unimodal'
-- loadParamsAndPackages(datasetType, modality) -- 'mir' or 'nus', 'I', or 'X'
-- resetGlobals()
-- loadVariableTrainingParams(baseLearningRate, baseWeightDecay, baseLearningRateDecay, mom)
-- loadModelAndOptimState() - set p.layerSizes if using custom text model
-- loadData() -- uses dataLoader.lua
-- optional: loadModelSnapshot -- from createModel.lua
-- trainAndEvaluate(numEpochs, batchSize)
-- /////////////////////////////////////////

function reloadAuxfPackage(pname)
  local pkg = 'auxf.' .. pname
  package.loaded[pkg] = nil
  require(pkg)
end

function runEverything()

    local datasetType = 'nus'
    local modality = 'X'

    loadParamsAndPackages(datasetType, modality)
    p.saveAccThreshold = 86
    resetGlobals()
    loadVariableTrainingParams()
    loadModelAndOptimState()
    loadData()
end

function tempTest()

    local dst = 'nus'
    local mod = 'X'
    loadParamsAndPackages(dst, mod)
    resetGlobals()
    loadVariableTrainingParams()
    loadModelAndOptimState()
    d = torch.load('/home/kjoslyn/kevin/Project/temp/d.t7')
end

function doSetLRForLayer(layerIdx, newLRMult)
    m.classifier:get(layerIdx):learningRate('weight', newLRMult)
    m.classifier:get(layerIdx):learningRate('bias', newLRMult)
end

function doSetWDForLayer(layerIdx, newWDMult)
    m.classifier:get(layerIdx):weightDecay('weight', newWDMult)
    m.classifier:get(layerIdx):weightDecay('bias', newWDMult)
end

function doChangeLRAndWDForLayer(layerName, newLR, newWD)

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
    o.optimState.learningRates = learningRates
    o.optimState.weightDecays = weightDecays
end

function changeLRAndWDForLayer(layerName, newLR, newWD)
    doChangeLRAndWDForLayer(layerName, newLR, newWD)
    setOptimStateLRAndWD(p.baseLearningRate, p.baseWeightDecay)
end

function loadParamsAndPackages(datasetType, modality, plotNumEpochs)

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
    s = {} -- stats for plotting

    matio = require 'matio'

    g.userPath = os.getenv("HOME") -- will be either '/home/kejosl' or '/home/kjoslyn'

    local snapshotDatasetDir
    if datasetType == 'mir' then
        p.numClasses = 24
        p.tagDim = 1075 -- Change tagDim if using PCA
        snapshotDatasetDir = '/mirflickr'
        g.datasetPath = g.userPath .. '/datasets/mirflickr/'
        g.evalTrainAccEpochs = 5
    elseif datasetType == 'nus' then
        p.numClasses = 21
        p.tagDim = 1000
        snapshotDatasetDir = '/nuswide'
        g.datasetPath = g.userPath .. '/datasets/nuswide/'
        g.evalTrainAccEpochs = 1
    else
        print("Error: Unrecognized datasetType!! Should be mir or nus")
    end
    p.datasetType = datasetType
    p.modality = modality
    p.batchSize = 100

    g.snapshotDir = g.userPath .. '/kevin/Project/snapshots' .. snapshotDatasetDir

    if plotNumEpochs then
        g.plotNumEpochs = plotNumEpochs
    elseif datasetType == 'nus' and modality == 'I' then
        g.plotNumEpochs = 1
    else
        g.plotNumEpochs = 5
    end

    if p.modality == 'X' then
        g.evalTrainAccEpochs = 10
    end

    p.topLearningRate = 0.01
    p.lastLayerLearningRate = 0.1

    p.topWeightDecay =  1 -- .01
    p.lastLayerWeightDecay =  1 -- .02
end

function resetGlobals()
    s.accIdx = 0
    s.pastNAcc = torch.Tensor(g.plotNumEpochs)
    s.avgDataAcc = torch.Tensor()
    s.maxDataAcc = torch.Tensor()
    s.minDataAcc = torch.Tensor()
    s.pastNLoss = torch.Tensor(g.plotNumEpochs)
    s.avgDataLoss = torch.Tensor()
    s.maxDataLoss = torch.Tensor()
    s.minDataLoss = torch.Tensor()
    g.plotStartEpoch = 1
    g.numEpochsCompleted = 0
end

function loadVariableTrainingParams(baseLearningRate, baseWeightDecay, baseLearningRateDecay, mom)

    -- //////////// Set training params - NOT USED IN mainUni.lua
    -- TODO: Fix image-text disparity
    -- LearningRate
    if p.modality == 'I' then
        p.baseLearningRate = baseLearningRate or 1e-4
    else
        p.baseLearningRate = baseLearningRate or .1
    end

    -- WeightDecay
    p.baseWeightDecay = baseWeightDecay or 0 -- 1e-4

    -- LearningRateDecay
    p.baseLearningRateDecay = baseLearningRateDecay or 0

    -- Momentum
    p.baseMomentum = mom or 0.9
end

function loadModelAndOptimState()

    -- //////////// Load image / text model
    if p.modality == 'I' then
    --    m.classifier = getImageModel()
       m.classifier = getImageModelImageNetPretrained(1e3)
       m.imageClassifier = m.classifier
    elseif p.modality == 'X' then
       m.classifier = getUntrainedTextModel(p.layerSizes) -- p.layerSizes would be set in mainUni.lua
       m.textClassifier = m.classifier
    else
       print('Error in unimodal.lua: Unrecognized modality')
    end

    m.classifier = m.classifier:cuda()

    o.optimState = {
        learningRate = p.baseLearningRate,
        learningRateDecay = p.baseLearningRateDecay,
        momentum = p.baseMomentum
    }

    if p.modality == 'I' then
        doChangeLRAndWDForLayer('top', p.topLearningRate, p.topWeightDecay)
        doChangeLRAndWDForLayer('last', p.lastLayerLearningRate, p.lastLayerWeightDecay)
    end
    setOptimStateLRAndWD(p.baseLearningRate, p.baseWeightDecay)

    o.params, o.gradParams = m.classifier:getParameters() 

    -- criterion = nn.MultiLabelSoftMarginCriterion()
    -- criterion = nn.MSECriterion()
    criterion = nn.BCECriterion()
    if p.modality == 'I' or p.noSizeAverage then
        criterion.sizeAverage = false -- TODO: Why does this work better for image modality?
    end
    criterion = criterion:cuda()
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

function doOneEpoch()
    doSGD()
    return doEvals()
end

function doSGD()

    local params = o.params
    local gradParams = o.gradParams
    local optimState = o.optimState

    m.classifier:training()

    if not g.epochTimer then
        g.epochTimer = torch.Timer()
    end

    g.epochTimer:reset()
    g.epochTimer:resume()
    -- shuffle at each epoch
    perm = torch.randperm(Ntrain):long()
    totalLoss = 0
    totNumIncorrect = 0

    numBatches = math.ceil(Ntrain / p.batchSize)

    for batchNum = 0, numBatches - 1 do

        trainBatch = getBatch(batchNum, p.batchSize, perm)
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
        optim.sgd(feval, params, optimState)

        -- collectgarbage()
    end
    g.epochTimer:stop()

    g.numEpochsCompleted = g.numEpochsCompleted + 1
end

function doEvals()

    local epoch = g.numEpochsCompleted

    collectgarbage()
    m.classifier:evaluate()
    avgLoss = totalLoss / numBatches
    statsPrint("Epoch " .. epoch .. ": avg loss = " .. avgLoss, g.sf)
    avgNumIncorrect = totNumIncorrect / Ntrain
    statsPrint("Epoch " .. epoch .. ": avg num incorrect = " .. avgNumIncorrect, g.sf)
    local valClassAcc
    if d.valset then
        valClassAcc = getClassAccuracy(d.valset.data, d.valset.label:cuda())
        statsPrint(string.format("Epoch %d: Val set class accuracy = %.2f", epoch, valClassAcc), g.sf)
    end
    if d.trainset and epoch % g.evalTrainAccEpochs == 0 then
        local trainClassAcc = getClassAccuracy(d.trainset.data, d.trainset.label:cuda())
        statsPrint(string.format("Epoch %d: Train set class accuracy = %.2f", epoch, trainClassAcc), g.sf)
    end
    print(string.format('Epoch time: %.2f seconds\n', g.epochTimer:time().real))
    m.classifier:training()

    s.accIdx = s.accIdx + 1
    s.pastNAcc[s.accIdx] = valClassAcc
    s.pastNLoss[s.accIdx] = math.min(avgLoss * 100, 100) -- Scale loss to fit in the same y axes as accuracy
    if s.accIdx % g.plotNumEpochs == 0 then
        s.avgDataAcc = s.avgDataAcc:cat(torch.Tensor({s.pastNAcc:mean()}))
        s.maxDataAcc = s.maxDataAcc:cat(torch.Tensor({s.pastNAcc:max()}))
        s.minDataAcc = s.minDataAcc:cat(torch.Tensor({s.pastNAcc:min()}))
        s.avgDataLoss = s.avgDataLoss:cat(torch.Tensor({s.pastNLoss:mean()}))
        s.maxDataLoss = s.maxDataLoss:cat(torch.Tensor({s.pastNLoss:max()}))
        s.minDataLoss = s.minDataLoss:cat(torch.Tensor({s.pastNLoss:min()}))
        if not g.skipPlot then
            -- TODO: I'm not sure how to make this work again
            -- if not g.plotStartEpoch then
            --     g.plotStartEpoch = epoch
            -- end
            plotClassAccAndLoss(epoch, true)
        end
        s.accIdx = 0
    end

    -- local save
    -- if p.modality == 'I' then
    --     save = epoch % 10 == 0
    -- else
    --     -- save = epoch == 200 or epoch == 300 or epoch == 500 or epoch == 750 or epoch == 1000 or epoch == 1500 or epoch == 2000
    --     save = false
    -- end

    -- if save then
    -- -- if epoch == 300 or epoch == 500 or epoch == 750 or epoch == 900 or epoch == 1000 then
    --     local snapshotFilename
    --     if g.snapshotFilename then
    --         snapshotFilename = g.snapshotFilename
    --     else
    --         local date = os.date("*t", os.time())
    --         snapshotFilename = date.month .. "_" .. date.day .. "_" .. date.hour .. "_" .. date.min
    --     end
    --     snapshotFilename = snapshotFilename .. '_snapshot_epoch_' .. epoch
    --     saveSnapshot(snapshotFilename, params, gradParams)
    -- end

    return avgLoss, valClassAcc
end

function trainAndEvaluate(numEpochs, batchSize, lr, mom, wd)

    assert(p.saveAccThreshold)

    -- ///////////////////////
    -- This section that uses the input parameters is not used in mainUni.lua since it sets the parameters
    -- before the run
    if batchSize then
        p.batchSize = batchSize
    end

    -- lr and wd parameters should only be used when not setting different learning rates for different layers
    -- (i.e. text modality)
    if lr or wd then
        local lr = lr or p.baseLearningRate
        local wd = wd or p.baseWeightDecay
        setOptimStateLRAndWD(lr, wd)
    end
    if mom then
        o.optimState.momentum = mom
    end
    -- /////////////////////////

    local bestValAcc = 0
    local bestValAccEpoch = 0

    for epoch = 1, numEpochs do
        local loss, valAcc = doOneEpoch()

        if valAcc > bestValAcc then
            bestValAcc = valAcc
            bestValAccEpoch = epoch
            if valAcc > p.saveAccThreshold then
                if not g.snapshotFilename then
                    local date = os.date("*t", os.time())
                    g.snapshotFilename = date.month .. "_" .. date.day .. "_" .. date.hour .. "_" .. date.min
                end
                local name = g.snapshotFilename .. '_best'
                saveSnapshot(name, o.params, o.gradParams)
            end
        end
    end
end

function saveSnapshot(filename, params, gradParams)
    local modalityDir
    if p.modality == 'I' then
        modalityDir = 'imageNet'
    else
        modalityDir = 'textNet'
    end
    local snapshot = {}
    snapshot.params = params
    snapshot.gradParams = gradParams
    -- snapshot.params, snapshot.gparams = m.classifier:getParameters()
    snapshot.s = s
    torch.save(g.snapshotDir .. '/' .. modalityDir .. '/' .. filename .. '.t7', snapshot)
end