
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

function loadParamsAndPackages()

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
  iterationsPerEpoch = 100

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
end

function reloadAuxfPackage(pname)
  local pkg = 'auxf.' .. pname
  package.loaded[pkg] = nil
  require(pkg)
end

function loadFullModel(modelType, lrMultForHashLayer, loadSiameseModels)

  loadParamsAndPackages()

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
    imageSiameseModel = getSiameseHasher(imageHasher)
    textSiameseModel = getSiameseHasher(textHasher)
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

  if not pos_pairs_full then
      pos_pairs_full, neg_pairs_full, trainImages, trainTexts, valImages, valTexts, pos_pairs_image, neg_pairs_image, pos_pairs_text, neg_pairs_text = pickSubset(true)
  end

end -- end loadData()

function runEvals()

  fullModel:evaluate()

  imageAccuracy = calcClassAccuracyForModality(I)
  textAccuracy = calcClassAccuracyForModality(X)
  statsPrint(string.format('Image Classification Acc: %.2f', imageAccuracy), sf, sfv)
  statsPrint(string.format('Text Classification Acc: %.2f', textAccuracy), sf, sfv)

  batchTextClassAcc = calcClassAccuracy(trainBatch.data[X], trainBatch.label[X])
  batchImageClassAcc = calcClassAccuracy(trainBatch.data[I], trainBatch.label[I])-- TODO: This is not very useful because it is only for the last batch in the epoch
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

  if not paramsAndOptimStatePrepared then
    criterion = getCriterion()
    getAndShareParameters(modality)
  end

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

function getAndShareParameters(modality)

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

function doOneEpochOnModality(modality, evalEpoch, logResults)

  -- The label tensor will be the same for each batch
  batch_sim_label = torch.Tensor(posExamplesPerBatch):fill(1)
  batch_sim_label = batch_sim_label:cat(torch.Tensor(negExamplesPerBatch):fill(0))
  batch_sim_label = torch.CudaByteTensor(totNumExamplesPerBatch):copy(batch_sim_label)
  batch_sim_label_for_loss_fixed = torch.CudaTensor(totNumExamplesPerBatch):copy(batch_sim_label) * L

  local model, params, gradParams, optimState, pos_pairs, neg_pairs

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
    print('Error: unrecognized modality in doOneEpochIntraModal')
  end

  trainBatch = {}

  if trainOnOneBatch == 1 then
    print("**************WARNING- Training on one batch only")
    trainBatch = getBatch(pos_pairs, neg_pairs, modality)
  end

  model:training()

  epochLoss = 0

  for batchNum = 0, numBatches - 1 do

      if trainOnOneBatch == 0 then
          trainBatch = getBatch(pos_pairs, neg_pairs, modality)
      end

      function feval(x)
          -- get new parameters
          if x ~= params then -- TODO: This is never happening
            params:copy(x) 
          end         

          input = trainBatch.data
          if sim_label_type == 'fixed' then
            target = batch_sim_label_for_loss_fixed
          elseif sim_label_type == 'variable' then
            target = trainBatch.batch_sim_label_for_loss
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

          return loss, gradParams
      end
      optim.sgd(feval, params, optimState)

  end

  statsPrint("=== " .. modality .. " ===Epoch " .. evalEpoch, sf, sfv)
  -- calcAndPrintHammingAccuracy(trainBatch, batch_sim_label, sfv) -- TODO: This is not very useful because it is only for the last batch in the epoch
  statsPrint(string.format("Avg Loss this epoch = %.2f", epochLoss / numBatches), sf, sfv)

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