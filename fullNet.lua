
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

function loadParamsAndPackages(datasetType, iterationsPerEpoch)

  if not nn then
    loadStandardPackages()
  end

  -- Global variable containers
  p = {} -- parameters
  d = {} -- data
  m = {} -- models
  g = {} -- other global variables
  o = {} -- optimStates and model parameters

  -- Variable Parameters
  -- numEpochs = 200 -- 416 is max number without truncating an epoch. This is now an input parameter to trainAndEvaluate
  -- p.lrMultForHashLayer = 1e4 -- 1e4, 1e5, etc
  -- p.modelType = 'gr' -- 'hgr', 'fc', 'hfc'
  p.L = 8
  p.k = 4
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

  -- Fixed Parameters
  I = 1 -- Table index for image modality - This is its own global variable
  X = 2 -- Table index for text modality - This is its own global variable
  g.filePath = '/home/kjoslyn/kevin/' -- server
  g.snapshotDir = '/home/kjoslyn/kevin/Project/snapshots' .. snapshotDatasetDir

  reloadAuxfPackage('pickSubset')
  reloadAuxfPackage('evaluate')
  reloadAuxfPackage('dataLoader')
  reloadAuxfPackage('batchLoader')
  reloadAuxfPackage('createModel')
  reloadAuxfPackage('map')

  g.y_loss = torch.Tensor()
  g.y_cross = torch.Tensor()
  g.y_b1 = torch.Tensor()
  g.y_b2 = torch.Tensor()
  g.y_q1 = torch.Tensor()
  g.y_q2 = torch.Tensor()
  g.y_xit = torch.Tensor()
  g.y_ixt = torch.Tensor()
  g.y_xiv = torch.Tensor()
  g.y_ixv = torch.Tensor()
end

function reloadAuxfPackage(pname)
  local pkg = 'auxf.' .. pname
  package.loaded[pkg] = nil
  require(pkg)
end

function loadFullModel(modelType, lrMultForHashLayer, loadSiameseModels)

  collectgarbage()

  p.modelType = modelType

  m.imageClassifier, m.imageHasher = getImageModelForFullNet(p.L, p.k, modelType, lrMultForHashLayer)
  m.textClassifier, m.textHasher = getTextModelForFullNet(p.L, p.k, modelType, lrMultForHashLayer)
  m.fullModel = createCombinedModel(m.imageHasher, m.textHasher)

  if loadSiameseModels then
    m.imageSiameseModel = createCombinedModel(m.imageHasher, m.imageHasher:clone())
    m.textSiameseModel = createCombinedModel(m.textHasher, m.textHasher:clone())
  end
end

function loadData() 

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

  d.trainset[I].data, d.trainset[I].label = d.dataset:getBySplit('training', 'I', 1, d.dataset:sizeTrain())
  d.trainset[X].data, d.trainset[X].label = d.dataset:getBySplit('training', 'X', 1, d.dataset:sizeTrain())

  d.valset[I].data, d.valset[I].label = d.dataset:getBySplit('val', 'I', 1, d.dataset:sizeVal())
  d.valset[X].data, d.valset[X].label = d.dataset:getBySplit('val', 'X', 1, d.dataset:sizeVal())

  d.testset[I].data, d.testset[I].label = d.dataset:getBySplit('query', 'I', 1, d.dataset:sizeTest())
  d.testset[X].data, d.testset[X].label = d.dataset:getBySplit('query', 'X', 1, d.dataset:sizeTest())

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

  local imageAccuracy = getClassAccuracyForModality(I)
  local textAccuracy = getClassAccuracyForModality(X)
  statsPrint(string.format('Train Image Classification Acc: %.2f', imageAccuracy), g.sf, g.sfv)
  statsPrint(string.format('Train Text Classification Acc: %.2f', textAccuracy), g.sf, g.sfv)

  local batchTextClassAcc = getClassAccuracy(trainBatch.data[X], trainBatch.label[X])
  local batchImageClassAcc = getClassAccuracy(trainBatch.data[I], trainBatch.label[I])-- TODO: This is not very useful because it is only for the last batch in the epoch
  statsPrint(string.format("Batch Text Classification Acc = %.2f", batchTextClassAcc), g.sfv)
  statsPrint(string.format("Batch Image Classification Acc = %.2f", batchImageClassAcc), g.sfv)

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
  local classesTo = {'training','pretraining','val'}
  local IXt = calcMAP(I, X, 'training', classesTo, true)
  local XIt = calcMAP(X, I, 'training', classesTo, true)
  local IXv = calcMAP(I, X, 'val', classesTo, true)
  local XIv = calcMAP(X, I, 'val', classesTo, true)
  local IIt = calcMAP(I, I, 'training', classesTo, true)
  local XXt = calcMAP(X, X, 'training', classesTo, true)
  local IIv = calcMAP(I, I, 'val', classesTo, true)
  local XXv = calcMAP(X, X, 'val', classesTo, true)

  statsPrint(string.format("X -> I train MAP = %.2f", XIt), g.sf, g.sfv)
  statsPrint(string.format("I -> X train MAP = %.2f", IXt), g.sf, g.sfv)
  statsPrint(string.format("X -> X train MAP = %.2f", XXt), g.sf, g.sfv)
  statsPrint(string.format("I -> I train MAP = %.2f", IIt), g.sf, g.sfv)
  statsPrint(string.format("X -> I val MAP = %.2f", XIv), g.sf, g.sfv)
  statsPrint(string.format("I -> X val MAP = %.2f", IXv), g.sf, g.sfv)
  statsPrint(string.format("X -> X val MAP = %.2f", XXv), g.sf, g.sfv)
  statsPrint(string.format("I -> I val MAP = %.2f", IIv), g.sf, g.sfv)

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

    if evalInterval and g.overallEpoch % evalInterval == 0 then

      local IXt, XIt, IXv, XIv = runEvals()

      g.y_ixt = g.y_ixt:cat(torch.Tensor({IXt}))
      g.y_xit = g.y_xit:cat(torch.Tensor({XIt}))
      g.y_ixv = g.y_ixv:cat(torch.Tensor({IXv}))
      g.y_xiv = g.y_xiv:cat(torch.Tensor({XIv}))
    elseif g.overallEpoch < evalInterval then
      g.y_ixt = g.y_ixt:cat(torch.Tensor({.5}))
      g.y_xit = g.y_xit:cat(torch.Tensor({.5}))
      g.y_ixv = g.y_ixv:cat(torch.Tensor({.5}))
      g.y_xiv = g.y_xiv:cat(torch.Tensor({.5}))
    else
      g.y_ixt = g.y_ixt:cat(torch.Tensor({g.y_ixt[g.overallEpoch - 1]}))
      g.y_xit = g.y_xit:cat(torch.Tensor({g.y_xit[g.overallEpoch - 1]}))
      g.y_ixv = g.y_ixv:cat(torch.Tensor({g.y_ixv[g.overallEpoch - 1]}))
      g.y_xiv = g.y_xiv:cat(torch.Tensor({g.y_xiv[g.overallEpoch - 1]}))
    end

    if plot then
      plotCrossModalLoss(g.overallEpoch)
    end
  end

  if logResults then
    io.close(g.sf)
    -- io.close(g.sfv)
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

function changeLearningRateForClassifier(lrMult)

  if not classifierWeightIndices then
    classifierWeightIndices = o.optimState_full.learningRates:eq(1)
    hashLayerIndices = o.optimState_full.learningRates:ne(1)
  end
  o.optimState_full.learningRates[classifierWeightIndices] = lrMult

end

function changeLearningRateForHashLayer(lrMult)

  -- Image
  if p.modelType == 'hfc' then
    m.fullModel:get(1):get(1):get(2):get(1):get(23):get(2):get(1):learningRate('weight', lrMult)
    m.fullModel:get(1):get(1):get(2):get(1):get(23):get(2):get(1):learningRate('bias', lrMult)
    m.fullModel:get(1):get(1):get(2):get(1):get(23):get(2):get(2):learningRate('weight', lrMult)
    m.fullModel:get(1):get(1):get(2):get(1):get(23):get(2):get(2):learningRate('bias', lrMult)
  elseif p.modelType == 'hgr' then
    m.fullModel:get(1):get(1):get(2):get(1):get(23):get(2):get(1):learningRate('weight', lrMult)
    m.fullModel:get(1):get(1):get(2):get(1):get(23):get(2):get(1):learningRate('bias', lrMult)
    m.fullModel:get(1):get(1):get(2):get(1):get(23):get(2):get(4):get(1):learningRate('weight', lrMult)
    m.fullModel:get(1):get(1):get(2):get(1):get(23):get(2):get(4):get(1):learningRate('bias', lrMult)
  end

  -- Text
  -- 2 hidden layer text model
  -- m.fullModel:get(1):get(2):get(2):get(1):get(10):get(2):get(1):learningRate('weight', lrMult)
  -- m.fullModel:get(1):get(2):get(2):get(1):get(10):get(2):get(1):learningRate('bias', lrMult)
  -- m.fullModel:get(1):get(2):get(2):get(1):get(10):get(2):get(2):learningRate('weight', lrMult)
  -- m.fullModel:get(1):get(2):get(2):get(1):get(10):get(2):get(2):learningRate('bias', lrMult)
  -- 1 hidden layer text model
  if p.modelType == 'hfc' then
    m.fullModel:get(1):get(2):get(2):get(1):get(7):get(2):get(1):learningRate('weight', lrMult)
    m.fullModel:get(1):get(2):get(2):get(1):get(7):get(2):get(1):learningRate('bias', lrMult)
    m.fullModel:get(1):get(2):get(2):get(1):get(7):get(2):get(2):learningRate('weight', lrMult)
    m.fullModel:get(1):get(2):get(2):get(1):get(7):get(2):get(2):learningRate('bias', lrMult)
  elseif p.modelType == 'hgr' then
    m.fullModel:get(1):get(2):get(2):get(1):get(7):get(2):get(1):learningRate('weight', lrMult)
    m.fullModel:get(1):get(2):get(2):get(1):get(7):get(2):get(1):learningRate('bias', lrMult)
    m.fullModel:get(1):get(2):get(2):get(1):get(7):get(2):get(4):get(1):learningRate('weight', lrMult)
    m.fullModel:get(1):get(2):get(2):get(1):get(7):get(2):get(4):get(1):learningRate('bias', lrMult)
  end

  local learningRates, weightDecays = m.fullModel:getOptimConfig(p.baseLearningRate, p.baseWeightDecay)
  o.optimState_full.learningRates = learningRates
  o.optimState_full.weightDecays = weightDecays
end

function changeWeightDecayForHashLayer(wdMult)

  -- Image
  m.fullModel:get(1):get(1):get(2):get(1):get(23):get(2):get(1):weightDecay('weight', wdMult)
  m.fullModel:get(1):get(1):get(2):get(1):get(23):get(2):get(1):weightDecay('bias', wdMult)
  m.fullModel:get(1):get(1):get(2):get(1):get(23):get(2):get(2):weightDecay('weight', wdMult)
  m.fullModel:get(1):get(1):get(2):get(1):get(23):get(2):get(2):weightDecay('bias', wdMult)

  -- Text
  -- 2 hidden layer text model
  -- m.fullModel:get(1):get(2):get(2):get(1):get(10):get(2):get(1):weightDecay('weight', wdMult)
  -- m.fullModel:get(1):get(2):get(2):get(1):get(10):get(2):get(1):weightDecay('bias', wdMult)
  -- m.fullModel:get(1):get(2):get(2):get(1):get(10):get(2):get(2):weightDecay('weight', wdMult)
  -- m.fullModel:get(1):get(2):get(2):get(1):get(10):get(2):get(2):weightDecay('bias', wdMult)
  -- 1 hidden layer text model
  m.fullModel:get(1):get(2):get(2):get(1):get(7):get(2):get(1):weightDecay('weight', wdMult)
  m.fullModel:get(1):get(2):get(2):get(1):get(7):get(2):get(1):weightDecay('bias', wdMult)
  m.fullModel:get(1):get(2):get(2):get(1):get(7):get(2):get(2):weightDecay('weight', wdMult)
  m.fullModel:get(1):get(2):get(2):get(1):get(7):get(2):get(2):weightDecay('bias', wdMult)

  local learningRates, weightDecays = m.fullModel:getOptimConfig(p.baseLearningRate, p.baseWeightDecay)
  o.optimState_full.learningRates = learningRates
  o.optimState_full.weightDecays = weightDecays
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
  g.y_loss = addPointToPlotLine(g.y_loss, avgEpochLoss)
  g.y_cross = addPointToPlotLine(g.y_cross, crossModalEpochLoss)
  g.y_b1 = addPointToPlotLine(g.y_b1, b1Loss)
  g.y_b2 = addPointToPlotLine(g.y_b2, b2Loss)
  g.y_q1 = addPointToPlotLine(g.y_q1, q1Loss)
  g.y_q2 = addPointToPlotLine(g.y_q2, q2Loss)

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

function doRunEverything()
  runEverything('mir',50,'hgr',5e4,'C',1,.015,0)
end

-- function runEverything(datasetType, iterationsPerEpoch, modelType, lrMultForHashLayer, kNum, modality, simWeight, balanceWeight, quantWeight)
function runEverything(datasetType, iterationsPerEpoch, modelType, lrMultForHashLayer, modality, simWeight, balanceWeight, quantWeight)

  loadParamsAndPackages(datasetType, iterationsPerEpoch)
  loadFullModel(modelType, lrMultForHashLayer)
  loadData()
  -- loadTrainAndValSubsets(kNum)
  getOptimStateAndShareParameters(modality)
  -- changeWeightDecayForHashLayer(1e2)
  doGetCriterion(simWeight, balanceWeight, quantWeight)
end

function saveSnapshot(filename)
    local snapshot = {}
    snapshot.params, snapshot.gparams = m.fullModel:getParameters()
    snapshot.g = g
    torch.save(g.snapshotDir .. '/' .. filename .. '.t7', snapshot)
end