
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

end -- end loadPackages()

function loadParamsAndPackages(iterationsPerEpoch)

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
  p.baseLearningRate = 1e-6
  p.baseLearningRateDecay = 0 -- 1e-3
  p.baseMomentum = 0 -- .9
  p.baseWeightDecay = 0
  p.posExamplesPerBatch = 25 -- 20
  p.negExamplesPerBatch = 75 -- 100
  -- p.iterationsPerEpoch = 100
  p.iterationsPerEpoch = iterationsPerEpoch
  p.kFoldSplitSize = 500
  p.kFoldNumSplits = 5

  -- These are inferred from above
  local posExamplesPerEpoch = p.posExamplesPerBatch*p.iterationsPerEpoch
  local negExamplesPerEpoch = p.negExamplesPerBatch*p.iterationsPerEpoch
  local epochSize = posExamplesPerEpoch + negExamplesPerEpoch
  p.batchSize = p.posExamplesPerBatch + p.negExamplesPerBatch
  p.numBatches = epochSize / p.batchSize

  -- Variable Boolean Parameters (1 or 0)
  p.trainOnOneBatch = 0

  -- Fixed Parameters
  I = 1 -- Table index for image modality - This is its own global variable
  X = 2 -- Table index for text modality - This is its own global variable
  g.filePath = '/home/kjoslyn/kevin/' -- server
  g.snapshotDir = '/home/kjoslyn/kevin/Project/snapshots'

  reloadAuxfPackage('pickSubset')
  reloadAuxfPackage('evaluate')
  reloadAuxfPackage('dataLoader')
  reloadAuxfPackage('batchLoader')
  reloadAuxfPackage('createModel')
  reloadAuxfPackage('map')
end

function reloadAuxfPackage(pname)
  local pkg = 'auxf.' .. pname
  package.loaded[pkg] = nil
  require(pkg)
end

function loadFullModel(modelType, lrMultForHashLayer, loadSiameseModels)

  collectgarbage()

  m.imageClassifier, m.imageHasher = getImageModelForFullNet(p.L, p.k, modelType, lrMultForHashLayer)
  m.textClassifier, m.textHasher = getTextModelForFullNet(p.L, p.k, modelType, lrMultForHashLayer)
  m.fullModel = createCombinedModel(m.imageHasher, m.textHasher)

  if loadSiameseModels then
    m.imageSiameseModel = createCombinedModel(m.imageHasher, m.imageHasher:clone())
    m.textSiameseModel = createCombinedModel(m.textHasher, m.textHasher:clone())
  end
end

function loadData() 

  if not d.trainset then
      d.trainset = {}
      d.testset = {}

      -- d.train_images and test_images each contain data and label fields
      local train_images, test_images = getImageData()
      d.trainset[I] = train_images.data
      d.testset[I] = test_images.data

      d.train_labels_image = train_images.label
      d.test_labels_image = test_images.label

      d.trainset[X], d.testset[X], d.train_labels_text, d.test_labels_text = getTextData()
  end

  -- if not d.pos_pairs_full then
  --     d.pos_pairs_full, d.neg_pairs_full, d.trainImages, d.trainTexts, d.valImages, d.valTexts, d.pos_pairs_image, d.neg_pairs_image, d.pos_pairs_text, d.neg_pairs_text = pickSubset(true)
  -- end

end -- end loadData()

function loadTrainAndValSubsets(kNum)

  if not kNum then
      d.pos_pairs_full, d.neg_pairs_full, d.trainImages, d.trainTexts, d.valImages, d.valTexts, d.pos_pairs_image, d.neg_pairs_image, d.pos_pairs_text, d.neg_pairs_text = pickSubset(true)
  else
      if not d.kFold_images then
        pickKFoldSubset(p.kFoldSplitSize, p.kFoldNumSplits, true)
      end
      d.pos_pairs_full, d.neg_pairs_full, d.trainImages, d.trainTexts, d.valImages, d.valTexts = getKFoldSplit(kNum)
      d.kNumLoaded = kNum
  end
end

function runEvals()

  m.fullModel:evaluate()

  hbc, stdev_image, stdev_text = getHashCodeBitCounts(d.trainset)
  statsPrint(string.format("Stdev I = %.2f", stdev_image), g.sf, g.sfv)
  statsPrint(string.format("Stdev X = %.2f", stdev_text), g.sf, g.sfv)

  statsPrint(string.format("Avg 0.5Dist I = %.3f", getSoftMaxAvgDistFromOneHalf(I)), g.sf, g.sfv)
  statsPrint(string.format("Avg 0.5Dist X = %.3f", getSoftMaxAvgDistFromOneHalf(X)), g.sf, g.sfv)

  local imageAccuracy = getClassAccuracyForModality(I)
  local textAccuracy = getClassAccuracyForModality(X)
  statsPrint(string.format('Image Classification Acc: %.2f', imageAccuracy), g.sf, g.sfv)
  statsPrint(string.format('Text Classification Acc: %.2f', textAccuracy), g.sf, g.sfv)

  local batchTextClassAcc = getClassAccuracy(trainBatch.data[X], trainBatch.label[X])
  local batchImageClassAcc = getClassAccuracy(trainBatch.data[I], trainBatch.label[I])-- TODO: This is not very useful because it is only for the last batch in the epoch
  statsPrint(string.format("Batch Text Classification Acc = %.2f", batchTextClassAcc), g.sfv)
  statsPrint(string.format("Batch Image Classification Acc = %.2f", batchImageClassAcc), g.sfv)

  local IXt = calcMAP(I, X, 'train')
  local XIt = calcMAP(X, I, 'train')
  local IXv = calcMAP(I, X, 'val')
  local XIv = calcMAP(X, I, 'val')
  local IIt = calcMAP(I, I, 'train')
  local XXt = calcMAP(X, X, 'train')
  local IIv = calcMAP(I, I, 'val')
  local XXv = calcMAP(X, X, 'val')

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

function trainAndEvaluate(modality, numEpochs, evalInterval, arg1, arg2)

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
    doOneEpochOnModality(modality, epoch, logResults)

    if epoch % evalInterval == 0 then
      runEvals()
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
    hashLayerIndices = o.optimState_full.learningRates:neq(1)
  end
  o.optimState_full.learningRates[classifierWeightIndices] = lrMult

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
  local trainSize = d.trainset[1]:size(1)
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

function doOneEpochOnModality(modality, evalEpoch, logResults)

  local model, params, gradParams, optimState, pos_pairs, neg_pairs = getModalitySpecifics(modality)

  trainBatch = {}

  if p.trainOnOneBatch == 1 then
    print("**************WARNING- Training on one batch only")
    trainBatch = getBatch(pos_pairs, neg_pairs, modality)
  end

  model:training()

  local epochLoss = 0
  local criterionLosses = torch.Tensor(#m.criterion.criterions):fill(0)

  for batchNum = 0, p.numBatches - 1 do

      if p.trainOnOneBatch == 0 then
          trainBatch = getBatch(pos_pairs, neg_pairs, modality)
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

  statsPrint(string.format("=== %s ===Epoch %d", modality, torch.round(optimState.evalCounter / p.iterationsPerEpoch)), g.sf, g.sfv)
  -- calcAndPrintHammingAccuracy(trainBatch, d.batch_sim_label, g.sfv) -- TODO: This is not very useful because it is only for the last batch in the epoch
  local avgEpochLoss = epochLoss / p.numBatches
  local crossModalEpochLoss = criterionLosses[1] / p.numBatches
  statsPrint(string.format("Avg Loss this epoch = %.2f", avgEpochLoss), g.sf, g.sfv)
  statsPrint(string.format("Cross Avg Loss this epoch = %.2f", crossModalEpochLoss), g.sf, g.sfv)
  statsPrint(string.format("Bal1 Avg Loss this epoch = %.2f", criterionLosses[2] / p.numBatches), g.sf, g.sfv)
  statsPrint(string.format("Bal2 Avg Loss this epoch = %.2f", criterionLosses[3] / p.numBatches), g.sf, g.sfv)
  statsPrint(string.format("Quant1 Avg Loss this epoch = %.2f", criterionLosses[4] / p.numBatches), g.sf, g.sfv)
  statsPrint(string.format("Quant2 Avg Loss this epoch = %.2f", criterionLosses[5] / p.numBatches), g.sf, g.sfv)

  if logResults and evalEpoch % 50 == 0 then
      local snapshotFile = g.snapshotDir .. "/snapshot_epoch_" .. epoch .. ".t7" 
      local snapshot = {}
      snapshot.params = params
      -- snapshot.params = torch.CudaTensor(params:size()):copy(params)
      if evalEpoch % 100 == 0 then
          -- snapshot.gparams = torch.CudaTensor(gradParams:size()):copy(gradParams)
          snapshot.gparams = gradParams
      end
      torch.save(snapshotFile, snapshot)
  end

  return avgEpochLoss, crossModalEpochLoss
end

function runEverything(iterationsPerEpoch, modelType, lrMultForHashLayer, kNum, modality, simWeight, balanceWeight, quantWeight)

  loadParamsAndPackages(iterationsPerEpoch)
  loadFullModel(modelType, lrMultForHashLayer)
  loadData()
  loadTrainAndValSubsets(kNum)
  getOptimStateAndShareParameters(modality)
  doGetCriterion(simWeight, balanceWeight, quantWeight)

end