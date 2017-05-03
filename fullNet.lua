
-- //////////////////////////////////////////
-- Typical flow:
-- require 'fullNet'
-- loadFullModel()
-- loadData()
-- optional: loadModelSnapshot() -- createModel.lua
-- trainAndEvaluate() or runEvals()
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

  -- Variable Parameters
  -- numEpochs = 200 -- 416 is max number without truncating an epoch. This is now an input parameter to trainAndEvaluate
  g_lrMultForHashLayer = 1e4 -- 1e4, 1e5, etc
  g_modelType = 'gr' -- 'hgr', 'fc', 'hfc'
  L = 8
  k = 4
  sim_label_type = 'fixed' -- 'variable'
  hashLayerSize = L * k
  baseLearningRate = 1e-6
  baseLearningRateDecay = 0 -- 1e-3
  baseMomentum = 0 -- .9
  baseWeightDecay = 0
  posExamplesPerBatch = 25 -- 20
  negExamplesPerBatch = 75 -- 100
  -- iterationsPerEpoch = 100
  kFoldSplitSize = 500
  kFoldNumSplits = 5

  -- These are inferred from above
  posExamplesPerEpoch = posExamplesPerBatch*iterationsPerEpoch
  negExamplesPerEpoch = negExamplesPerBatch*iterationsPerEpoch
  epochSize = posExamplesPerEpoch + negExamplesPerEpoch
  totNumExamplesPerBatch = posExamplesPerBatch + negExamplesPerBatch
  numBatches = epochSize / totNumExamplesPerBatch

  -- Variable Boolean Parameters (1 or 0)
  trainOnOneBatch   = 0
  loadModelFromFile = 0
  saveModelToFile   = 0

  -- Fixed Parameters
  I = 1 -- Table index for image modality
  X = 2 -- Table index for text modality
  filePath = '/home/kjoslyn/kevin/' -- server
  snapshotDir = '/home/kjoslyn/kevin/Project/snapshots'

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

  if not modelType then
    modelType = g_modelType
  end
  if not lrMultForHashLayer then
    lrMultForHashLayer = g_lrMultForHashLayer
  end

  imageClassifier, imageHasher = getImageModelForFullNet(L, k, modelType, lrMultForHashLayer)
  textClassifier, textHasher = getTextModelForFullNet(L, k, modelType, lrMultForHashLayer)
  fullModel = createCombinedModel(imageHasher, textHasher)

  if loadSiameseModels then
    imageSiameseModel = createCombinedModel(imageHasher, imageHasher:clone())
    textSiameseModel = createCombinedModel(textHasher, textHasher:clone())
  end
end

function loadData() 

  if not trainset then
      trainset = {}
      testset = {}

      -- train_images and test_images each contain data and label fields
      train_images, test_images = getImageData()
      trainset[I] = train_images.data
      testset[I] = test_images.data

      train_labels_image = train_images.label
      test_labels_image = test_images.label

      trainset[X], testset[X], train_labels_text, test_labels_text = getTextData()
  end

  -- if not pos_pairs_full then
  --     pos_pairs_full, neg_pairs_full, trainImages, trainTexts, valImages, valTexts, pos_pairs_image, neg_pairs_image, pos_pairs_text, neg_pairs_text = pickSubset(true)
  -- end

end -- end loadData()

function loadTrainAndValSubsets(kNum)

  if not kNum then
      pos_pairs_full, neg_pairs_full, trainImages, trainTexts, valImages, valTexts, pos_pairs_image, neg_pairs_image, pos_pairs_text, neg_pairs_text = pickSubset(true)
  else
      if not kFold_images then
        pickKFoldSubset(kFoldSplitSize, kFoldNumSplits, true)
      end
      pos_pairs_full, neg_pairs_full, trainImages, trainTexts, valImages, valTexts = getKFoldSplit(kNum)
  end
end

function runEvals()

  fullModel:evaluate()

  hbc, stdev_image, stdev_text = getHashCodeBitCounts(trainset)
  print(string.format("Stdev I = %.2f", stdev_image))
  print(string.format("Stdev X = %.2f", stdev_text))

  print(string.format("Avg 0.5Dist I = %.3f", getSoftMaxAvgDistFromOneHalf(I)))
  print(string.format("Avg 0.5Dist X = %.3f", getSoftMaxAvgDistFromOneHalf(X)))

  imageAccuracy = getClassAccuracyForModality(I)
  textAccuracy = getClassAccuracyForModality(X)
  statsPrint(string.format('Image Classification Acc: %.2f', imageAccuracy), sf, sfv)
  statsPrint(string.format('Text Classification Acc: %.2f', textAccuracy), sf, sfv)

  batchTextClassAcc = getClassAccuracy(trainBatch.data[X], trainBatch.label[X])
  batchImageClassAcc = getClassAccuracy(trainBatch.data[I], trainBatch.label[I])-- TODO: This is not very useful because it is only for the last batch in the epoch
  statsPrint(string.format("Batch Text Classification Acc = %.2f", batchTextClassAcc), sfv)
  statsPrint(string.format("Batch Image Classification Acc = %.2f", batchImageClassAcc), sfv)

  IXt = calcMAP(I, X, 'train')
  XIt = calcMAP(X, I, 'train')
  IXv = calcMAP(I, X, 'val')
  XIv = calcMAP(X, I, 'val')
  IIt = calcMAP(I, I, 'train')
  XXt = calcMAP(X, X, 'train')
  IIv = calcMAP(I, I, 'val')
  XXv = calcMAP(X, X, 'val')

  statsPrint(string.format("X -> I train MAP = %.2f", XIt), sf, sfv)
  statsPrint(string.format("I -> X train MAP = %.2f", IXt), sf, sfv)
  statsPrint(string.format("X -> X train MAP = %.2f", XXt), sf, sfv)
  statsPrint(string.format("I -> I train MAP = %.2f", IIt), sf, sfv)
  statsPrint(string.format("X -> I val MAP = %.2f", XIv), sf, sfv)
  statsPrint(string.format("I -> X val MAP = %.2f", IXv), sf, sfv)
  statsPrint(string.format("X -> X val MAP = %.2f", XXv), sf, sfv)
  statsPrint(string.format("I -> I val MAP = %.2f", IIv), sf, sfv)

end

function trainAndEvaluate(modality, numEpochs, evalInterval, arg1, arg2)

  local paramsAndOptimStatePrepared = arg1 and arg1 == 'skip' or arg2 and arg2 == 'skip'
  local logResults = arg1 and arg1 == 'log' or arg2 and arg2 == 'log'

  if logResults then
    local date = os.date("*t", os.time())
    local dateStr = date.month .. "_" .. date.day .. "_" .. date.hour .. "_" .. date.min
    sf = io.open(snapshotDir .. "/stats_" .. dateStr .. ".txt", "w")
    sfv = io.open(snapshotDir .. "/stats_verbose_" .. dateStr .. ".txt", "w")
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
    io.close(sf)
    io.close(sfv)
  end

end

function doGetCriterion(simWeight, balanceWeight, quantWeight)
  criterion = getCriterion(simWeight, balanceWeight, quantWeight)
end

function getOptimStateAndShareParameters(modality)

  collectgarbage()
  params_full = nil
  gradParams_full = nil
  optimState_full = nil
  params_image = nil
  gradParams_image = nil
  optimState_image = nil
  params_text = nil
  gradParams_text = nil
  optimState_text = nil
  
  if modality == 'C' or modality == 'A' then -- TODO: Implement 'A' modality

    print('***WARNING- Getting full model parameters, siamese weight sharing will be destroyed')
    params_full, gradParams_full = fullModel:getParameters() -- This destroys the weight sharing for the siamese models!

    local learningRates_full, weightDecays_full = fullModel:getOptimConfig(baseLearningRate, baseWeightDecay)

    optimState_full = {
          learningRate = baseLearningRate,
          learningRateDecay = baseLearningRateDecay,
          learningRates = learningRates_full,
          weightDecays = weightDecays_full,
          momentum = baseMomentum
    }

  end
  if modality == 'I' or modality == 'A' then

    params_image, gradParams_image = imageSiameseModel:getParameters()
    imageSiameseModel:get(1):get(2):share(imageSiameseModel:get(1):get(1), 'bias', 'weight', 'gradWeight', 'gradParams')

    local learningRates_image, weightDecays_image = imageSiameseModel:getOptimConfig(baseLearningRate, baseWeightDecay)

    optimState_image = {
          learningRate = baseLearningRate,
          learningRateDecay = baseLearningRateDecay,
          learningRates = learningRates_image,
          weightDecays = weightDecays_image,
          momentum = baseMomentum
    }

  end
  if modality == 'X' or modality == 'A' then

    params_text, gradParams_text = textSiameseModel:getParameters()
    textSiameseModel:get(1):get(2):share(textSiameseModel:get(1):get(1), 'bias', 'weight', 'gradWeight', 'gradParams')

    local learningRates_text, weightDecays_text = textSiameseModel:getOptimConfig(baseLearningRate, baseWeightDecay)

    optimState_text = {
          learningRate = baseLearningRate,
          learningRateDecay = baseLearningRateDecay,
          learningRates = learningRates_text,
          weightDecays = weightDecays_text,
          momentum = baseMomentum
    }

  end
end

function changeLearningRateForClassifier(lrMult)

  if not classifierWeightIndices then
    classifierWeightIndices = optimState_full.learningRates:eq(1)
    hashLayerIndices = optimState_full.learningRates:neq(1)
  end
  optimState_full.learningRates[classifierWeightIndices] = lrMult

end

function getModalitySpecifics(modality)

  if modality == 'X' then
    model = textSiameseModel
    params = params_text
    gradParams = gradParams_text
    optimState = optimState_text
    pos_pairs = pos_pairs_text
    neg_pairs = neg_pairs_text
  elseif modality == 'I' then
    model = imageSiameseModel
    params = params_image
    gradParams = gradParams_image
    optimState = optimState_image
    pos_pairs = pos_pairs_image
    neg_pairs = neg_pairs_image
  elseif modality == 'C' then
    model = fullModel
    params = params_full
    gradParams = gradParams_full
    optimState = optimState_full
    pos_pairs = pos_pairs_full
    neg_pairs = neg_pairs_full
  else
    print('Error: unrecognized modality in getModalitySpecifics')
  end

  return model, params, gradParams, optimState, pos_pairs, neg_pairs
end

function getInputAndTarget(modality, trainBatch)

  local batchSize = trainBatch.data[1]:size(1)
  local trainSize = trainset[1]:size(1)
  local trainEstimatorConst = trainSize / batchSize
  -- beta_im_pre = torch.sum(imPred, 1):view(-1):mul(trainEstimatorConst)
  -- beta_te_pre = torch.sum(tePred, 1):view(-1):mul(trainEstimatorConst)
  -- local alpha = trainSize / k
  local pred1, pred2
  if modality == 'I' or modality == 'C' then
    imPred = imageHasher:forward(trainBatch.data[I])
    pred1 = imPred
  end
  if modality == 'X' or modality == 'C' then
    tePred = textHasher:forward(trainBatch.data[X])
    pred2 = tePred
  end
  if modality == 'X' then
    pred1 = tePred
  elseif modality == 'I' then
    pred2 = imPred
  end

  beta1 = torch.sum(pred1, 1):view(-1)
  beta2 = torch.sum(pred2, 1):view(-1)
  local alpha = batchSize / k
  gamma1 = beta1 - 2*alpha
  gamma2 = beta2 - 2*alpha
  gamma1 = torch.expand(gamma1:resize(1,L*k), batchSize, L*k)
  gamma2 = torch.expand(gamma2:resize(1,L*k), batchSize, L*k)
  local bt = - L * (batchSize / k)
  balance_target = torch.CudaTensor(batchSize):fill(bt)

  -- beta1 = torch.sum(pred1, 1):view(-1) * (trainSize / batchSize)
  -- beta2 = torch.sum(pred2, 1):view(-1) * (trainSize / batchSize)
  -- local alpha = trainSize / k
  -- gamma1 = beta1 - 2*alpha
  -- gamma2 = beta2 - 2*alpha
  -- gamma1 = torch.expand(gamma1:resize(1,L*k), batchSize, L*k)
  -- gamma2 = torch.expand(gamma2:resize(1,L*k), batchSize, L*k)
  -- local bt = - L * (trainSize / k)
  -- balance_target = torch.CudaTensor(batchSize):fill(bt)

  quant_target = torch.CudaTensor(batchSize):fill(0.5*L*k)

  input = {}
  input[1] = trainBatch.data[1]
  input[2] = trainBatch.data[2]
  input[3] = gamma1
  input[4] = gamma2

  target = {}
  if sim_label_type == 'fixed' then
    target[1] = batch_sim_label_for_loss_fixed
  elseif sim_label_type == 'variable' then
    target[1] = trainBatch.batch_sim_label_for_loss
  end
  target[2] = balance_target
  target[3] = balance_target
  target[4] = quant_target
  target[5] = quant_target

  return input, target
end

function doOneEpochOnModality(modality, evalEpoch, logResults)

  -- The label tensor will be the same for each batch
  batch_sim_label = torch.Tensor(posExamplesPerBatch):fill(1)
  batch_sim_label = batch_sim_label:cat(torch.Tensor(negExamplesPerBatch):fill(0))
  batch_sim_label = torch.CudaByteTensor(totNumExamplesPerBatch):copy(batch_sim_label)
  batch_sim_label_for_loss_fixed = torch.CudaTensor(totNumExamplesPerBatch):copy(batch_sim_label) * L -- for MSECriterion
  -- batch_sim_label_for_loss_fixed = torch.CudaTensor(totNumExamplesPerBatch):copy(batch_sim_label) -- for BCECriterion only

  local model, params, gradParams, optimState, pos_pairs, neg_pairs = getModalitySpecifics(modality)

  trainBatch = {}

  if trainOnOneBatch == 1 then
    print("**************WARNING- Training on one batch only")
    trainBatch = getBatch(pos_pairs, neg_pairs, modality)
  end

  model:training()

  epochLoss = 0
  crossModalEpochLoss = 0
  balance1EpochLoss = 0
  balance2EpochLoss = 0
  quant1EpochLoss = 0
  quant2EpochLoss = 0

  for batchNum = 0, numBatches - 1 do

      if trainOnOneBatch == 0 then
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
          local loss = criterion:forward(output, target)
          local dloss_doutput = criterion:backward(output, target)
          model:backward(input, dloss_doutput)

          gradParams:div(inputSize)
          loss = loss/inputSize

          -- Stats
          epochLoss = epochLoss + loss
          crossModalEpochLoss = crossModalEpochLoss + critSim:forward(output[1], target[1])/inputSize
          balance1EpochLoss = balance1EpochLoss + critBalanceIm:forward(output[2], target[2])/inputSize
          balance2EpochLoss = balance2EpochLoss + critBalanceTe:forward(output[3], target[3])/inputSize
          quant1EpochLoss = quant1EpochLoss + critQuantIm:forward(output[4], target[4])/inputSize
          quant2EpochLoss = quant2EpochLoss + critQuantTe:forward(output[5], target[5])/inputSize

          return loss, gradParams
      end
      optim.sgd(feval, params, optimState)

  end

  statsPrint(string.format("=== %s ===Epoch %d", modality, torch.round(optimState.evalCounter / iterationsPerEpoch)), sf, sfv)
  -- calcAndPrintHammingAccuracy(trainBatch, batch_sim_label, sfv) -- TODO: This is not very useful because it is only for the last batch in the epoch
  statsPrint(string.format("Avg Loss this epoch = %.2f", epochLoss / numBatches), sf, sfv)
  statsPrint(string.format("Cross Avg Loss this epoch = %.2f", crossModalEpochLoss / numBatches), sf, sfv)
  statsPrint(string.format("Bal1 Avg Loss this epoch = %.2f", balance1EpochLoss / numBatches), sf, sfv)
  statsPrint(string.format("Bal2 Avg Loss this epoch = %.2f", balance2EpochLoss / numBatches), sf, sfv)
  statsPrint(string.format("Quant1 Avg Loss this epoch = %.2f", quant1EpochLoss / numBatches), sf, sfv)
  statsPrint(string.format("Quant2 Avg Loss this epoch = %.2f", quant2EpochLoss / numBatches), sf, sfv)

  if logResults and evalEpoch % 50 == 0 then
      local snapshotFile = snapshotDir .. "/snapshot_epoch_" .. epoch .. ".t7" 
      local snapshot = {}
      snapshot.params = params
      -- snapshot.params = torch.CudaTensor(params:size()):copy(params)
      if evalEpoch % 100 == 0 then
          -- snapshot.gparams = torch.CudaTensor(gradParams:size()):copy(gradParams)
          snapshot.gparams = gradParams
      end
      torch.save(snapshotFile, snapshot)
  end
end

function runEverything(modelType, lrMultForHashLayer, kNum, modality, simWeight, balanceWeight, quantWeight)

  loadParamsAndPackages()
  loadFullModel(modelType, lrMultForHashLayer)
  loadData()
  loadTrainAndValSubsets(kNum)
  getOptimStateAndShareParameters(modality)
  doGetCriterion(simWeight, balanceWeight, quantWeight)

end