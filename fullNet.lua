
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

function loadCustomPackages()

  package.loaded.pickSubset = nil
  package.loaded.evaluate = nil
  package.loaded.dataLoader = nil
  package.loaded.batchLoader = nil
  package.loaded.createModel = nil
  require 'pickSubset'
  require 'auxf.evaluate'
  require 'auxf.dataLoader'
  require 'batchLoader'
  require 'createModel'
end

function loadModelSnapshot()

  snapshot2ndLevelDir = 'Lr5e4_5kquery_5kdatabase'
  snapshotFileName = 'snapshot_epoch_100.t7'

  snapshotFullPath = snapshotDir .. '/' .. snapshot2ndLevelDir .. '/' .. snapshotFileName
  snapshot = torch.load(snapshotFullPath)
  N = snapshot.params:size(1)

  params, gparams = model:getParameters()

  batchSize = 1e5

  local numBatches = torch.ceil(N / batchSize)
  for b = 0, numBatches - 1 do
      startIndex = b * batchSize + 1
      endIndex = math.min((b + 1) * batchSize, N)
      params[ {{ startIndex, endIndex }} ]:copy(snapshot.params[ {{ startIndex, endIndex }} ])
  end

end

function loadModelAndData() 

  if not nn then
    loadStandardPackages()
  end

  -- Variable Parameters
  numEpochs = 200 -- 416 is max number without truncating an epoch
  -- posExamplesPerEpoch = 1e4
  -- negExamplesPerEpoch = 5e4
  posExamplesPerEpoch = 20*100
  negExamplesPerEpoch = 100*100
  L = 8
  k = 4
  hashLayerSize = L * k
  baseLearningRate = 1e-6
  baseWeightDecay = 0
  lrMultForHashLayer = 1e3 -- 1e4, 1e5, etc
  posExamplesPerBatch = 20
  negExamplesPerBatch = 100

  -- These are inferred from above
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
  snapshotDir = '/home/kjoslyn/kevin/FullNet/snapshots'

  loadCustomPackages()

  if not model then
      if loadModelFromFile == 1 then
          print('***Loading model from file')
          modelData = torch.load('modelData.t7')
          imageClassifier = modelData.imageClassifier
          imageHasher = modelData.imageHasher
          textClassifier = modelData.textClassifier
          textHasher = modelData.textHasher
          model = modelData.combinedModel
      else
          imageClassifier, imageHasher = getImageModel()
          textClassifier, textHasher = getTextModel()
          model = createCombinedModel(imageHasher, textHasher)

          if saveModelToFile == 1 then
          print('***Saving model to file')
          modelData = {}
          modelData.imageClassifier = imageClassifier
          modelData.imageHasher = imageHasher
          modelData.textClassifier = textClassifier
          modelData.textHasher = textHasher
          modelData.combinedModel = model
          torch.save('modelData.t7', modelData)
          else
          print('***Skipping save model to file')
          end
      end
  end

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

  if not pos_pairs then
      pos_pairs, neg_pairs, p_size, n_size = pickSubset(true)
  end
  -- TODO: These are global variables that come from pickSubset. I want to return them instead but the compiler gives an error.
  trainImages = imageIdx:long()
  trainTexts = textIdx:long()

end -- end loadModelAndData()

function trainAndEvaluate()

  -- calcMAP(X, I, nil) -- TODO: Remove
  -- calcMAP(I, X, nil) -- TODO: Remove

  criterion = getCriterion()

  learningRates, weightDecays = model:getOptimConfig(baseLearningRate, baseWeightDecay)

  local optimState = {
        learningRate = baseLearningRate,
        learningRates = learningRates,
        weightDecays = weightDecays
        -- learningRateDecay = 1e-7
        -- learningRate = 1e-3,
        -- learningRateDecay = 1e-4
        -- weightDecay = 0.01,
        -- momentum = 0.9
  }

  params, gradParams = model:getParameters()

  -- The label tensor will be the same for each batch
  batch_sim_label = torch.Tensor(posExamplesPerBatch):fill(1)
  batch_sim_label = batch_sim_label:cat(torch.Tensor(negExamplesPerBatch):fill(0))
  batch_sim_label = torch.CudaByteTensor(totNumExamplesPerBatch):copy(batch_sim_label)
  batch_label_for_loss = torch.CudaTensor(totNumExamplesPerBatch):copy(batch_sim_label) * L

  -- -- POS ONLY The label tensor will be the same for each batch
  -- batch_sim_label = torch.Tensor(100):fill(1)
  -- batch_sim_label = torch.CudaByteTensor(100):copy(batch_sim_label)
  -- batch_label_for_loss = torch.CudaTensor(100):copy(batch_sim_label) * L

  -- -- NEG ONLY The label tensor will be the same for each batch
  -- batch_sim_label = torch.Tensor(100):fill(0)
  -- batch_sim_label = torch.CudaByteTensor(100):copy(batch_sim_label)
  -- batch_label_for_loss = torch.CudaTensor(100):copy(batch_sim_label)

  iterationsComplete = 0

  trainBatch = {}

  if trainOnOneBatch == 1 then
    print("**************WARNING- Training on one batch only")
    trainBatch = getBatch(pos_pairs, neg_pairs, p_size, n_size)
  end

  totalLoss = 0
  -- epochHistorySize = 5
  -- epochHistoryLoss = torch.Tensor(epochHistorySize):fill(0)

  date = os.date("*t", os.time())
  dateStr = date.month .. "_" .. date.day .. "_" .. date.hour .. "_" .. date.min
  sf = io.open(snapshotDir .. "/stats_" .. dateStr .. ".txt", "w")
  sfv = io.open(snapshotDir .. "/stats_verbose_" .. dateStr .. ".txt", "w")

  model:training()

  for epoch = 1, numEpochs  do

      epochLoss = 0

      for batchNum = 0, numBatches - 1 do

          if trainOnOneBatch == 0 then
            trainBatch = getBatch(pos_pairs, neg_pairs, p_size, n_size)
          end

          function feval(x)
              -- get new parameters
              if x ~= params then
                params:copy(x)
              end         

              input = trainBatch.data
              target = batch_label_for_loss
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
              -- epochHistoryLoss[(epoch % epochHistorySize) + 1] = loss
              totalLoss = totalLoss + loss

              return loss, gradParams
          end
          optim.sgd(feval, params, optimState)

          iterationsComplete = iterationsComplete + 1

          -- collectgarbage()
      end

      model:evaluate()
      -- epochPred = model:forward(trainBatch.data)

      -- batchTextToImageMAP = calcMAP(X, I, trainBatch)
      -- batchImageToTextMAP = calcMAP(I, X, trainBatch)
      if epoch % 5 == 0 then
          textToImageMAP = calcMAP(X, I, nil)
          imageToTextMAP = calcMAP(I, X, nil)

          imageAccuracy = calcClassAccuracyForModality(I)
          textAccuracy = calcClassAccuracyForModality(X)

          batchTextClassAcc = calcClassAccuracy(trainBatch.data[X], trainBatch.label[X])
          batchImageClassAcc = calcClassAccuracy(trainBatch.data[I], trainBatch.label[I])-- TODO: This is not very useful because it is only for the last batch in the epoch

          statsPrint("=====Epoch " .. epoch, sf, sfv)
          calcAndPrintHammingAccuracy(trainBatch, batch_sim_label, sfv) -- TODO: This is not very useful because it is only for the last batch in the epoch

          statsPrint(string.format("Avg Loss this epoch = %.2f", epochLoss / numBatches), sf, sfv)
          statsPrint(string.format("Avg Loss overall = %.2f", totalLoss / iterationsComplete), sf, sfv)

          -- statsPrint(string.format("Batch Text to Image MAP = %.2f", batchTextToImageMAP))
          -- statsPrint(string.format("Batch Image to Text MAP = %.2f", batchImageToTextMAP))
          statsPrint(string.format("Text to Image MAP = %.2f", textToImageMAP), sf, sfv)
          statsPrint(string.format("Image to Text MAP = %.2f", imageToTextMAP), sf, sfv)

          statsPrint(string.format('Image Classification Acc: %.2f', imageAccuracy), sf, sfv)
          statsPrint(string.format('Text Classification Acc: %.2f', textAccuracy), sf, sfv)

          statsPrint(string.format("Batch Text Classification Acc = %.2f", batchTextClassAcc), sfv)
          statsPrint(string.format("Batch Image Classification Acc = %.2f", batchImageClassAcc), sfv)
      end

      if epoch % 50 == 0 then
          local paramsToSave, gp = model:getParameters()
          local snapshotFile = snapshotDir .. "/snapshot_epoch_" .. epoch .. ".t7" 
          local snapshot = {}
          snapshot.params = paramsToSave
          if epoch % 100 == 0 then
              snapshot.gparams = gp
          end
          torch.save(snapshotFile, snapshot)
      end

      model:training()
  end

  io.close(sf)
  io.close(sfv)

  --[[
  model:evaluate()
  --]]

end -- end trainAndEvaluate()