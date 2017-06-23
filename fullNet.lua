
-- //////////////////////////////////////////
-- Typical flow:
-- /////////////////////////////////////////

function loadStandardPackages() 

  require 'nn'
  require 'loadcaffe' -- doesn't work on server
  require 'image'
  require 'optim'
  require 'nnlr'
  require 'cutorch'
  require 'cunn'
  require 'cudnn'
  require 'imagenetloader.dataset'
  gnuplot = require 'gnuplot'

end -- end loadPackages()

function loadParamsAndPackages(datasetType, iterationsPerEpoch, usePretrainedImageFeatures, L, k)

  if not nn then
    loadStandardPackages()
  end

  -- Global variable containers
  p = {} -- parameters
  d = {} -- data
  m = {} -- models
  g = {} -- other global variables
  o = {} -- optimStates and model parameters
  s = {} -- stats for plotting

  g.userPath = os.getenv("HOME") -- will be either '/home/kejosl' or '/home/kjoslyn'

  -- Variable Parameters
  -- numEpochs = 200 -- 416 is max number without truncating an epoch. This is now an input parameter to trainAndEvaluate
  -- p.lrMultForHashLayer = 1e4 -- 1e4, 1e5, etc
  -- p.modelType = 'gr' -- 'hgr', 'fc', 'hfc'

  -- In main.lua, these are params in pfs. Thus, loadParamsAndPackages may be called without these params in main.lua.
  p.L = L -- 8 
  p.k = k -- 4

  p.usePretrainedImageFeatures = usePretrainedImageFeatures
  p.sim_label_type = 'fixed' -- 'variable'
  p.baseLearningRate = 1e-6 -- 1e-6
  p.baseLearningRateDecay = 0 -- 1e-3
  p.baseMomentum = .9 -- .9
  p.baseWeightDecay = 0 -- 1e-6
  p.posExamplesPerBatch = 50 -- 25, 20
  p.negExamplesPerBatch = 150 -- 75, 100
  p.iterationsPerEpoch = iterationsPerEpoch -- 25, (50), 100
  p.kFoldSplitSize = 500
  p.kFoldNumSplits = 5

  -- These are inferred from above
  p.posExamplesPerEpoch = p.posExamplesPerBatch*p.iterationsPerEpoch
  p.negExamplesPerEpoch = p.negExamplesPerBatch*p.iterationsPerEpoch
  local epochSize = p.posExamplesPerEpoch + p.negExamplesPerEpoch
  p.batchSize = p.posExamplesPerBatch + p.negExamplesPerBatch
  p.numBatches = epochSize / p.batchSize

  -- Variable Boolean Parameters (1 or 0)
  p.trainOnOneBatch = 0

  -- Dataset
  local snapshotDatasetDir
  if datasetType == 'mir' then
      p.numClasses = 24
      p.tagDim = 1075
      snapshotDatasetDir = '/mirflickr'
      g.datasetPath = g.userPath .. '/datasets/mirflickr/'
  elseif datasetType == 'nus' then
      p.numClasses = 21
      p.tagDim = 1000
      snapshotDatasetDir = '/nuswide'
      g.datasetPath = g.userPath .. '/datasets/nuswide/'
  else
      print("Error: Unrecognized datasetType!! Should be mir or nus")
  end
  p.datasetType = datasetType

  -- Fixed Parameters
  I = 1 -- Table index for image modality - This is its own global variable
  X = 2 -- Table index for text modality - This is its own global variable
  g.filePath = g.userPath .. '/kevin/' -- server
  g.snapshotDir = g.userPath .. '/kevin/Project/snapshots' .. snapshotDatasetDir

  reloadAuxfPackage('pickSubset')
  reloadAuxfPackage('evaluate')
  reloadAuxfPackage('dataLoader')
  reloadAuxfPackage('batchLoader')
  reloadAuxfPackage('createModel')
  reloadAuxfPackage('map')

  resetGlobals()
end

function resetGlobals()
    s.y_loss = torch.Tensor()
    s.y_cross = torch.Tensor()
    s.y_b1 = torch.Tensor()
    s.y_b2 = torch.Tensor()
    s.y_q1 = torch.Tensor()
    s.y_q2 = torch.Tensor()
    s.y_xit = torch.Tensor()
    s.y_ixt = torch.Tensor()
    s.y_xiv = torch.Tensor()
    s.y_ixv = torch.Tensor()
    g.plotStartEpoch = 1
end

function reloadAuxfPackage(pname)
  local pkg = 'auxf.' .. pname
  package.loaded[pkg] = nil
  require(pkg)
end

function loadFullModel(modelType, XHlrMult, IHlrMult, XClrMult, IClrMult, loadSiameseModels, layerSizes)

  collectgarbage()

  p.modelType = modelType
  p.XHlrMult = XHlrMult
  p.IHlrMult = IHlrMult
  p.XClrMult = XClrMult
  p.IClrMult = IClrMult

  if not p.usePretrainedImageFeatures then
    m.imageClassifier, m.imageHasher = getImageModelForFullNet(p.L, p.k, modelType, IHlrMult, IClrMult)
  else
    m.imageHasher = getHashLayer(4096, modelType, p.L, p.k, IHlrMult)
  end
  m.textClassifier, m.textHasher = getTextModelForFullNet(p.L, p.k, modelType, XHlrMult, XClrMult, layerSizes)
  m.fullModel = createCombinedModel(m.imageHasher, m.textHasher)

  if loadSiameseModels then
    m.imageSiameseModel = createCombinedModel(m.imageHasher, m.imageHasher:clone())
    m.textSiameseModel = createCombinedModel(m.textHasher, m.textHasher:clone())
  end
end

function loadData() 

  if p.usePretrainedImageFeatures then
    local data = torch.load(g.datasetPath .. 'alexNetFeatures_deep.t7')
    d.trainset = data.trainset
    d.valset = data.valset
    d.testset = data.testset
    d.pretrainset = data.pretrainset
  else
    d.trainset = {}
    d.trainset[I] = {}
    d.trainset[X] = {}
    d.valset = {}
    d.valset[I] = {}
    d.valset[X] = {}
    d.testset = {}
    d.testset[I] = {}
    d.testset[X] = {}
    d.pretrainset = {}
    d.pretrainset[I] = {}
    d.pretrainset[X] = {}

    local imageRootPath = g.datasetPath .. 'ImageData'
    d.dataset = imageLoader{path=imageRootPath, sampleSize={3,227,227}, splitFolders={'training', 'pretraining', 'val', 'query'}}

    d.trainset[I].data, d.trainset[X].data, d.trainset[I].label = d.dataset:getBySplit('training', 'B', 1, d.dataset:sizeTrain())
    d.trainset[X].label = d.trainset[I].label

    d.valset[I].data, d.valset[X].data, d.valset[I].label = d.dataset:getBySplit('val', 'B', 1, d.dataset:sizeVal())
    d.valset[X].label = d.valset[I].label

    d.testset[I].data, d.testset[X].data, d.testset[I].label = d.dataset:getBySplit('query', 'B', 1, d.dataset:sizeTest())
    d.testset[X].label = d.testset[I].label
  end

  local pairs = torch.load(g.datasetPath .. 'crossModalPairs.t7')
  d.pos_pairs_full = pairs.pos_pairs
  d.neg_pairs_full = pairs.neg_pairs

end -- end loadData()

function runEvals()

  m.fullModel:evaluate()

  hbc, stdev_image, stdev_text = getHashCodeBitCounts(d.trainset)
  statsPrint(string.format("Stdev I = %.2f", stdev_image), g.sf, g.sfv)
  statsPrint(string.format("Stdev X = %.2f", stdev_text), g.sf, g.sfv)

  statsPrint(string.format("Avg 0.5Dist I = %.3f", getSoftMaxAvgDistFromOneHalf(I)), g.sf, g.sfv)
  statsPrint(string.format("Avg 0.5Dist X = %.3f", getSoftMaxAvgDistFromOneHalf(X)), g.sf, g.sfv)

  local textAccuracy = getClassAccuracyForModality(X)
  statsPrint(string.format('Train Text Classification Acc: %.2f', textAccuracy), g.sf, g.sfv)
  local batchTextClassAcc = getClassAccuracy(trainBatch.data[X], trainBatch.label[X])
  statsPrint(string.format("Batch Text Classification Acc = %.2f", batchTextClassAcc), g.sfv)

  if not p.usePretrainedImageFeatures then
    local imageAccuracy = getClassAccuracyForModality(I)
    statsPrint(string.format('Train Image Classification Acc: %.2f', imageAccuracy), g.sf, g.sfv)
    local batchImageClassAcc = getClassAccuracy(trainBatch.data[I], trainBatch.label[I])-- TODO: This is not very useful because it is only for the last batch in the epoch
    statsPrint(string.format("Batch Image Classification Acc = %.2f", batchImageClassAcc), g.sfv)
  end

  -- Clear codes to signal need to compute the hash codes again
  d.trainset[I].codes = nil
  d.trainset[X].codes = nil
  d.pretrainset[I].codes = nil
  d.pretrainset[X].codes = nil
  d.valset[I].codes = nil
  d.valset[X].codes = nil
  d.testset[I].codes = nil
  d.testset[X].codes = nil
  -- local IXt = calcMAP(I, X, 'training', 'training', true)
  -- local XIt = calcMAP(X, I, 'training', 'training', true)
  -- local IXv = calcMAP(I, X, 'val', 'val', true)
  -- local XIv = calcMAP(X, I, 'val', 'val', true)
  -- local IIt = calcMAP(I, I, 'training', 'training', true)
  -- local XXt = calcMAP(X, X, 'training', 'training', true)
  -- local IIv = calcMAP(I, I, 'val', 'val', true)
  -- local XXv = calcMAP(X, X, 'val', 'val', true)

  -- local classesTo
  -- if p.datasetType == 'mir' then
    classesTo = {'training','val','pretraining'}
  -- else
  --   classesTo = {'training','val'}
  -- end

  local IXt, IXt_time = calcMAP(I, X, 'training', classesTo, true)
  local XIt, XIt_time = calcMAP(X, I, 'training', classesTo, true)
  local IXv, IXv_time = calcMAP(I, X, 'val', classesTo, true)
  local XIv, XIv_time = calcMAP(X, I, 'val', classesTo, true)
  local IIt = calcMAP(I, I, 'training', classesTo, true)
  local XXt = calcMAP(X, X, 'training', classesTo, true)
  local IIv = calcMAP(I, I, 'val', classesTo, true)
  local XXv = calcMAP(X, X, 'val', classesTo, true)

  statsPrint(string.format("X -> I train MAP = %.3f", XIt), g.sf, g.sfv)
  statsPrint(string.format("I -> X train MAP = %.3f", IXt), g.sf, g.sfv)
  statsPrint(string.format("X -> X train MAP = %.3f", XXt), g.sf, g.sfv)
  statsPrint(string.format("I -> I train MAP = %.3f", IIt), g.sf, g.sfv)
  statsPrint(string.format("X -> I val MAP = %.3f", XIv), g.sf, g.sfv)
  statsPrint(string.format("I -> X val MAP = %.3f", IXv), g.sf, g.sfv)
  statsPrint(string.format("X -> X val MAP = %.3f", XXv), g.sf, g.sfv)
  statsPrint(string.format("I -> I val MAP = %.3f", IIv), g.sf, g.sfv)

  print(string.format("X -> I train MAP time = %.2f", XIt_time))
  print(string.format("I -> X train MAP time = %.2f", IXt_time))
  print(string.format("X -> I val MAP time = %.2f", XIv_time))
  print(string.format("I -> X val MAP time = %.2f", IXv_time))

  return IXt, XIt, IXv, XIv

end

function trainAndEvaluate(modality, numEpochs, evalInterval, plot, arg1, arg2)

  local paramsAndOptimStatePrepared = arg1 and arg1 == 'skip' or arg2 and arg2 == 'skip'
  local logResults = arg1 and arg1 == 'log' or arg2 and arg2 == 'log'

  if logResults then
    local date = os.date("*t", os.time())
    local dateStr = date.month .. "_" .. date.day .. "_" .. date.hour .. "_" .. date.min
    g.sf = io.open(g.snapshotDir .. "/stats_" .. dateStr .. ".txt", "w")
    -- g.sfv = io.open(g.snapshotDir .. "/stats_verbose_" .. dateStr .. ".txt", "w")
  end

  -- if not paramsAndOptimStatePrepared then
  --   getOptimStateAndShareParameters(modality)
  -- end

  for epoch = 1, numEpochs do

    doOneEpochOnModality(modality, logResults)

    -- if g.overallEpoch == 10 then
    --   changeLearningRateForHashLayer(1e4)
    -- elseif g.overallEpoch == 50 then
    --   changeLearningRateForHashLayer(5e3)
    -- end
    local IXt, XIt, IXv, XIv
    if g.overallEpoch % evalInterval == 0 then
      IXt, XIt, IXv, XIv = runEvals()
    end

    addPlotStats(g.overallEpoch, evalInterval, IXt, XIt, IXv, XIv)

    if plot and g.overallEpoch % evalInterval == 0 then
      plotCrossModalLoss(g.overallEpoch)
    end
  end

  if logResults then
    io.close(g.sf)
    -- io.close(g.sfv)
  end

end

function addPlotStats(epoch, evalInterval, IXt, XIt, IXv, XIv)

    if epoch % evalInterval == 0 then

      s.y_ixt = s.y_ixt:cat(torch.Tensor({IXt}))
      s.y_xit = s.y_xit:cat(torch.Tensor({XIt}))
      s.y_ixv = s.y_ixv:cat(torch.Tensor({IXv}))
      s.y_xiv = s.y_xiv:cat(torch.Tensor({XIv}))
    elseif epoch < evalInterval then
      s.y_ixt = s.y_ixt:cat(torch.Tensor({.5}))
      s.y_xit = s.y_xit:cat(torch.Tensor({.5}))
      s.y_ixv = s.y_ixv:cat(torch.Tensor({.5}))
      s.y_xiv = s.y_xiv:cat(torch.Tensor({.5}))
    else
      s.y_ixt = s.y_ixt:cat(torch.Tensor({s.y_ixt[epoch - 1]}))
      s.y_xit = s.y_xit:cat(torch.Tensor({s.y_xit[epoch - 1]}))
      s.y_ixv = s.y_ixv:cat(torch.Tensor({s.y_ixv[epoch - 1]}))
      s.y_xiv = s.y_xiv:cat(torch.Tensor({s.y_xiv[epoch - 1]}))
    end
end

function doGetCriterion(simWeight, balanceWeight, quantWeight)
  m.criterion = getCriterion(simWeight, balanceWeight, quantWeight)
end

function getOptimStateAndShareParameters(modality)

  -- TODO: Get rid of this?
  o = {}
  collectgarbage()
  
  if modality == 'C' or modality == 'A' then -- TODO: Implement 'A' modality

    print('***WARNING- Getting full model parameters, siamese weight sharing will be destroyed')
    o.params_full, o.gradParams_full = m.fullModel:getParameters() -- This destroys the weight sharing for the siamese models!

    local learningRates_full, weightDecays_full = m.fullModel:getOptimConfig(p.baseLearningRate, p.baseWeightDecay)

    o.optimState_full = {
          learningRate = p.baseLearningRate,
          learningRateDecay = p.baseLearningRateDecay,
          learningRates = learningRates_full,
          weightDecays = weightDecays_full,
          momentum = p.baseMomentum
    }

  end
  if modality == 'I' or modality == 'A' then

    o.params_image, o.gradParams_image = m.imageSiameseModel:getParameters()
    m.imageSiameseModel:get(1):get(2):share(m.imageSiameseModel:get(1):get(1), 'bias', 'weight', 'gradWeight', 'gradParams')

    local learningRates_image, weightDecays_image = m.imageSiameseModel:getOptimConfig(p.baseLearningRate, p.baseWeightDecay)

    o.optimState_image = {
          learningRate = p.baseLearningRate,
          learningRateDecay = p.baseLearningRateDecay,
          learningRates = learningRates_image,
          weightDecays = weightDecays_image,
          momentum = p.baseMomentum
    }

  end
  if modality == 'X' or modality == 'A' then

    o.params_text, o.gradParams_text = m.textSiameseModel:getParameters()
    m.textSiameseModel:get(1):get(2):share(m.textSiameseModel:get(1):get(1), 'bias', 'weight', 'gradWeight', 'gradParams')

    local learningRates_text, weightDecays_text = m.textSiameseModel:getOptimConfig(p.baseLearningRate, p.baseWeightDecay)

    o.optimState_text = {
          learningRate = p.baseLearningRate,
          learningRateDecay = p.baseLearningRateDecay,
          learningRates = learningRates_text,
          weightDecays = weightDecays_text,
          momentum = p.baseMomentum
    }

  end
end

function getClassAndHashLayerIndices()
  if not o.classifierWeightIndices then
    o.classifierWeightIndices = {}
    o.classifierWeightIndices[I] = o.optimState_full.learningRates:eq(p.IClrMult)
    o.classifierWeightIndices[X] = o.optimState_full.learningRates:eq(p.XClrMult)
    -- o.classifierWeightIndices = o.optimState_full.learningRates:eq(1)
    o.hashLayerIndices = {}
    o.hashLayerIndices[I] = o.optimState_full.learningRates:eq(p.IHlrMult)
    o.hashLayerIndices[X] = o.optimState_full.learningRates:eq(p.XHlrMult)
  end
end

function changeLearningRateForClassifier(lrMult, modality)
  getClassAndHashLayerIndices()
  -- o.optimState_full.learningRates[o.classifierWeightIndices] = lrMult
  if not modality or modality == 'X' then
    o.optimState_full.learningRates[o.classifierWeightIndices[X]] = lrMult
    p.XClrMult = lrMult
  end
  if not modality or modality == 'I' then
    o.optimState_full.learningRates[o.classifierWeightIndices[I]] = lrMult
    p.IClrMult = lrMult
  end
end

function changeLearningRateForHashLayer(lrMult, modality)
  getClassAndHashLayerIndices()
  if not modality or modality == 'X' then
    o.optimState_full.learningRates[o.hashLayerIndices[X]] = lrMult
    p.XHlrMult = lrMult
  end
  if not modality or modality == 'I' then
    o.optimState_full.learningRates[o.hashLayerIndices[I]] = lrMult
    p.IHlrMult = lrMult
  end
end

function changeWeightDecayForHashLayer(wdMult)
  getClassAndHashLayerIndices()
  o.optimState_full.weightDecays[o.hashLayerIndices] = wdMult
end

function getModalitySpecifics(modality)

  local model, params, gradParams, optimState, pos_pairs, neg_pairs

  if modality == 'X' then
    model = m.textSiameseModel
    params = o.params_text
    gradParams = o.gradParams_text
    optimState = o.optimState_text
    pos_pairs = d.pos_pairs_text
    neg_pairs = d.neg_pairs_text
  elseif modality == 'I' then
    model = m.imageSiameseModel
    params = o.params_image
    gradParams = o.gradParams_image
    optimState = o.optimState_image
    pos_pairs = d.pos_pairs_image
    neg_pairs = d.neg_pairs_image
  elseif modality == 'C' then
    model = m.fullModel
    params = o.params_full
    gradParams = o.gradParams_full
    optimState = o.optimState_full
    pos_pairs = d.pos_pairs_full
    neg_pairs = d.neg_pairs_full
  else
    print('Error: unrecognized modality in getModalitySpecifics')
  end

  return model, params, gradParams, optimState, pos_pairs, neg_pairs
end

function getInputAndTarget(modality, trainBatch)

  if p.sim_label_type == 'fixed' and not g.batch_sim_label_for_loss_fixed then
    -- The label tensor will be the same for each batch
    local batch_sim_label = torch.Tensor(p.posExamplesPerBatch):fill(1)
    batch_sim_label = batch_sim_label:cat(torch.Tensor(p.negExamplesPerBatch):fill(0))
    batch_sim_label = torch.CudaByteTensor(p.batchSize):copy(batch_sim_label)
    g.batch_sim_label_for_loss_fixed = torch.CudaTensor(p.batchSize):copy(batch_sim_label) * p.L -- for MSECriterion
    -- g.batch_sim_label_for_loss_fixed = torch.CudaTensor(p.batchSize):copy(batch_sim_label) -- for BCECriterion only
  end

  local batchSize = trainBatch.data[1]:size(1)
  local trainSize = d.trainset[1].data:size(1)
  local trainEstimatorConst = trainSize / batchSize
  -- beta_im_pre = torch.sum(imPred, 1):view(-1):mul(trainEstimatorConst)
  -- beta_te_pre = torch.sum(tePred, 1):view(-1):mul(trainEstimatorConst)
  -- local alpha = trainSize / p.k
  local pred1, pred2, imPred, tePred
  if modality == 'I' or modality == 'C' then
    imPred = m.imageHasher:forward(trainBatch.data[I])
    pred1 = imPred
  end
  if modality == 'X' or modality == 'C' then
    tePred = m.textHasher:forward(trainBatch.data[X])
    pred2 = tePred
  end
  if modality == 'X' then
    pred1 = tePred
  elseif modality == 'I' then
    pred2 = imPred
  end

  local beta1 = torch.sum(pred1, 1):view(-1)
  local beta2 = torch.sum(pred2, 1):view(-1)
  local alpha = batchSize / p.k
  local gamma1 = beta1 - 2*alpha
  local gamma2 = beta2 - 2*alpha
  local gamma1 = torch.expand(gamma1:resize(1,p.L*p.k), batchSize, p.L*p.k)
  local gamma2 = torch.expand(gamma2:resize(1,p.L*p.k), batchSize, p.L*p.k)
  local bt = - p.L * (batchSize / p.k)
  local balance_target = torch.CudaTensor(batchSize):fill(bt)

  -- beta1 = torch.sum(pred1, 1):view(-1) * (trainSize / batchSize)
  -- beta2 = torch.sum(pred2, 1):view(-1) * (trainSize / batchSize)
  -- local alpha = trainSize / p.k
  -- gamma1 = beta1 - 2*alpha
  -- gamma2 = beta2 - 2*alpha
  -- gamma1 = torch.expand(gamma1:resize(1,p.L*p.k), batchSize, p.L*p.k)
  -- gamma2 = torch.expand(gamma2:resize(1,p.L*p.k), batchSize, p.L*p.k)
  -- local bt = - p.L * (trainSize / p.k)
  -- balance_target = torch.CudaTensor(batchSize):fill(bt)

  local quant_target = torch.CudaTensor(batchSize):fill(0.5*p.L*p.k)

  local input = {}
  input[1] = trainBatch.data[1]
  input[2] = trainBatch.data[2]
  input[3] = gamma1
  input[4] = gamma2

  local target = {}
  if p.sim_label_type == 'fixed' then
    target[1] = g.batch_sim_label_for_loss_fixed
  elseif p.sim_label_type == 'variable' then
    target[1] = trainBatch.batch_sim_label_for_loss
  end
  target[2] = balance_target
  target[3] = balance_target
  target[4] = quant_target
  target[5] = quant_target

  return input, target
end

local function addPointToPlotLine(tensor, val)
  tensor = tensor:cat(torch.Tensor({val}))
  return tensor
end

function doOneEpochOnModality(modality, logResults)

  local model, params, gradParams, optimState, pos_pairs, neg_pairs = getModalitySpecifics(modality)

  trainBatch = {}

  local pos_perm, neg_perm
  if p.trainOnOneBatch == 1 then
    print("**************WARNING- Training on one batch only")
    trainBatch = getBatch_old(pos_pairs, neg_pairs, modality)
  else
    pos_perm = torch.randperm(d.pos_pairs_full:size(1))[{ { 1, p.posExamplesPerEpoch } }]:long()
    neg_perm = torch.randperm(d.neg_pairs_full:size(1))[{ { 1, p.negExamplesPerEpoch } }]:long()
  end

  model:training()

  local epochLoss = 0
  local criterionLosses = torch.Tensor(#m.criterion.criterions):fill(0)

  for batchNum = 0, p.numBatches - 1 do

      if p.trainOnOneBatch == 0 then
          trainBatch = getBatch(batchNum, pos_pairs, neg_pairs, modality, pos_perm, neg_perm)
      end

      input, target = getInputAndTarget(modality, trainBatch)
      
      function feval(x)
          -- get new parameters
          if x ~= params then -- TODO: This is never happening
            params:copy(x) 
          end         

          inputSize = input[1]:size(1)

          gradParams:zero()

          output = model:forward(input)
          local loss = m.criterion:forward(output, target)
          local dloss_doutput = m.criterion:backward(output, target)
          model:backward(input, dloss_doutput)

          gradParams:div(inputSize)
          loss = loss/inputSize

          -- Stats
          epochLoss = epochLoss + loss
          for i = 1, criterionLosses:size(1) do
            criterionLosses[i] = criterionLosses[i] + m.criterion.criterions[i]:forward(output[i], target[i])/inputSize
          end

          return loss, gradParams
      end
      optim.sgd(feval, params, optimState)
      
      collectgarbage()

  end

  g.overallEpoch = torch.round(optimState.evalCounter / p.iterationsPerEpoch) -- TODO: Why round?

  statsPrint(string.format("=== %s ===Epoch %d", modality, g.overallEpoch), g.sf, g.sfv)
  -- calcAndPrintHammingAccuracy(trainBatch, d.batch_sim_label, g.sfv) -- TODO: This is not very useful because it is only for the last batch in the epoch
  local avgEpochLoss = epochLoss / p.numBatches
  local crossModalEpochLoss = criterionLosses[1] / p.numBatches
  local b1Loss = criterionLosses[2] / p.numBatches
  local b2Loss = criterionLosses[3] / p.numBatches
  local q1Loss = criterionLosses[4] / p.numBatches
  local q2Loss = criterionLosses[5] / p.numBatches
  statsPrint(string.format("Avg Loss this epoch = %.2f", avgEpochLoss), g.sf, g.sfv)
  statsPrint(string.format("Cross Avg Loss this epoch = %.2f", crossModalEpochLoss), g.sf, g.sfv)
  statsPrint(string.format("Bal1 Avg Loss this epoch = %.2f", b1Loss), g.sf, g.sfv)
  statsPrint(string.format("Bal2 Avg Loss this epoch = %.2f", b2Loss), g.sf, g.sfv)
  statsPrint(string.format("Quant1 Avg Loss this epoch = %.2f", q1Loss), g.sf, g.sfv)
  statsPrint(string.format("Quant2 Avg Loss this epoch = %.2f", q2Loss), g.sf, g.sfv)
  s.y_loss = addPointToPlotLine(s.y_loss, avgEpochLoss)
  s.y_cross = addPointToPlotLine(s.y_cross, crossModalEpochLoss)
  s.y_b1 = addPointToPlotLine(s.y_b1, b1Loss)
  s.y_b2 = addPointToPlotLine(s.y_b2, b2Loss)
  s.y_q1 = addPointToPlotLine(s.y_q1, q1Loss)
  s.y_q2 = addPointToPlotLine(s.y_q2, q2Loss)

  if logResults and g.overallEpoch % 50 == 0 then
      local snapshotFile = g.snapshotDir .. "/snapshot_epoch_" .. g.overallEpoch .. ".t7" 
      local snapshot = {}
      snapshot.params = params
      -- snapshot.params = torch.CudaTensor(params:size()):copy(params)
      if g.overallEpoch % 100 == 0 then
          -- snapshot.gparams = torch.CudaTensor(gradParams:size()):copy(gradParams)
          snapshot.gparams = gradParams
      end
      torch.save(snapshotFile, snapshot)
  end

  return avgEpochLoss, crossModalEpochLoss
end

function runEverything()

  -- These are the hardcoded variable params. Need to reload this file every time one changes.

  local datasetType = 'mir'
  local iterationsPerEpoch = 50
  local usePretrainedImageFeatures = false
  local L = 16
  local k = 4
  local modelType = 'hfc'
  local lrMultForHashLayer = 5e4
  local layerSizes = { 2048, 2048, 2048 }
  -- local layerSizes = { 't', 2048 }
  local modality = 'C'
  local simWeight = 1
  local balanceWeight = 0.015
  local quantWeight = 0.25

  local XHlrMult = lrMultForHashLayer
  local IHlrMult = lrMultForHashLayer
  local XClrMult = 1
  local IClrMult = 1

  loadParamsAndPackages(datasetType, iterationsPerEpoch, usePretrainedImageFeatures, L, k)
  -- p.baseMomentum = 0
  loadFullModel(modelType, XHlrMult, IHlrMult, XClrMult, IClrMult, false, layerSizes)
  loadData()
  -- loadTrainAndValSubsets(kNum)
  getOptimStateAndShareParameters(modality)
  -- changeWeightDecayForHashLayer(1e2)
  doGetCriterion(simWeight, balanceWeight, quantWeight)

  -- For illustration only
  -- trainAndEvaluate(modality, numEpochs, evalInterval, plot, arg1, arg2)
end

function saveSnapshot(filename, params, gradParams)
    local snapshot = {}
    -- snapshot.params, snapshot.gparams = m.fullModel:getParameters()
    snapshot.params = params
    snapshot.gradParams = gradParams
    snapshot.s = s
    torch.save(g.snapshotDir .. '/' .. filename .. '.t7', snapshot)
end

function getDistAndSimFromSnapshotsInDir(sPath)
  print('Warning: changing g.snapshotDir to ' .. sPath)
  g.snapshotDir = sPath -- This must be done for loadModelSnapshot to work properly
  -- Get filenames only, not full path
  local sDir = io.popen('find ' .. sPath .. '/*_bestAvg.t7 -type f -exec basename {} \\;')
  for fn in sDir:lines() do 
    loadModelSnapshot(m.fullModel, nil, fn)
    m.fullModel:evaluate()
    prepareTestMAPs(fn)
  end
end

-- TODO: Make these test and not val
function prepareTestMAPs(fn)

    local classesTo
    -- if p.datasetType == 'mir' then
        classesTo = {'training','val','pretraining'}
    -- else
    --     classesTo = {'training','val'}
    -- end

    d.trainset[I].codes = nil
    d.trainset[X].codes = nil
    d.pretrainset[I].codes = nil
    d.pretrainset[X].codes = nil
    d.valset[I].codes = nil
    d.valset[X].codes = nil
    d.testset[I].codes = nil
    d.testset[X].codes = nil

    local ixv_name = fn .. '_DS_data_IX_bestAvg.mat'
    local IXv = calcMAP(I, X, 'val', classesTo, true, ixv_name)

    local xiv_name = fn .. '_DS_data_XI_bestAvg.mat'
    local XIv = calcMAP(X, I, 'val', classesTo, true, xiv_name)
end