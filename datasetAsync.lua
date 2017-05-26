require 'torch'
local threads = require 'threads'
threads.Threads.serialization('threads.sharedserialize')

local status,tds = pcall(require, 'tds')
tds = status and tds or nil
assert(status)

local ffi = require 'ffi'
local dir = require 'pl.dir'
local tablex = require 'pl.tablex'
local argcheck = require 'argcheck'
require 'sys'
local gm = require 'graphicsmagick'

local dataset = torch.class('imageLoader')

local initcheck = argcheck{
   pack=true,
   help=[[
     A dataset class for images in a flat folder structure (folder-name is class-name).
     Optimized for extremely large datasets (upwards of 14 million images).
     Tested only on Linux (as it uses command-line linux utilities to scale up to 14 million+ images)
]],
   {check=function(path) 
       local files = dir.getfiles(path)
       local labelFileFound = false
       local avgStdFileFound = false
       for k,filepath in ipairs(files) do
          if string.match(filepath, 'labels.mat') then
             labelFileFound = true
          end
          if string.match(filepath, 'avgStdevTrainSet.t7') then
             avgStdFileFound = true
          end
       end
          
       return labelFileFound and avgStdFileFound
   end,
    name="path",
    type="string",
    help="Path of directory with images, containing folders \'training\', \'val\', \'query\', \'pretraining\', and avgStdevTrainSet.t7 and labels.mat"}, 

   {name="sampleSize",
    type="table",
    help="a consistent sample size to resize the images"},

   {name="splitFolders",
    type="table",
    help="examples: training, val, query, pretraining"},

   {name="samplingMode",
    type="string",
    help="Sampling mode: random | balanced ",
    default = "balanced"},

   {name="verbose",
    type="boolean",
    help="Verbose mode during initialization",
    default = false},

   {name="loadSize",
    type="table",
    help="a size to load the images to, initially",
    opt = true},

   {name="sampleHookTrain",
    type="function",
    help="applied to sample during training(ex: for lighting jitter). "
       .. "It takes the image path as input",
    opt = true},

   {name="sampleHookTest",
    type="function", 
    help="applied to sample during testing",
    opt = true}
}

function dataset:__init(...)
   
   -- argcheck
   local args = initcheck(...)
   print(args)
   for k,v in pairs(args) do self[k] = v end   

   if not self.loadSize then self.loadSize = self.sampleSize; end

   if not self.sampleHookTrain then self.sampleHookTrain = self.defaultSampleHook end
   if not self.sampleHookTest then self.sampleHookTest = self.defaultSampleHook end

   -- TODO: Handle this better
   nThreads = 8
   pool = threads.Threads(nThreads)

   -- Load labels and channel avgs and stdevs
   if not matio then
      matio = require 'matio'
   end

   local labels = matio.load(self.path .. '/labels.mat')
   self.labels = labels.labels

   local tags = matio.load(self.path .. '/tags.mat')
   self.tags = tags.tags

--    local avgStd = matio.load(path .. '/avgStdevTrainSet.mat')
   local avgStd = torch.load(self.path .. '/avgStdevTrainSet.t7')
   self.channelAvgs = avgStd.means
   self.channelStds = avgStd.stdev

   -- find class names
   self.classes = {}
   -- get list of unique class names in path folder (should be 'training', 'val', 'query', 'pretraining', 
   -- also store the directory paths per class
   -- for each class, 
   local classPaths = {}
   local dirs = dir.getdirectories(self.path);
   for k,dirpath in ipairs(dirs) do
       local class = paths.basename(dirpath)
       -- class will be one of "training", "val", "query", and "pretraining"
       -- only use the class folder if it was specified in splitFolders
       local splitIdx = tablex.find(self.splitFolders, class)
       if splitIdx then
          local idx = tablex.find(self.classes, class)
          if not idx then
              table.insert(self.classes, class)
              idx = #self.classes
              classPaths[idx] = {}
          end
          if not tablex.find(classPaths[idx], dirpath) then
             table.insert(classPaths[idx], dirpath);
          end
       end
   end
   
   self.classIndices = {}
   for k,v in ipairs(self.classes) do
      self.classIndices[v] = k
   end
   
   -- define command-line tools, try your best to maintain OSX compatibility
   local wc = 'wc'
   local cut = 'cut'
   local find = 'find'
   if jit.os == 'OSX' then
      wc = 'gwc'
      cut = 'gcut'
      find = 'gfind'
   end
   ----------------------------------------------------------------------
   -- Options for the GNU find command
   local extensionList = {'jpg', 'png','JPG','PNG','JPEG', 'ppm', 'PPM', 'bmp', 'BMP'}
--    local extensionList = {'jpg'}
   local findOptions = '-iname "*.' .. extensionList[1] .. '"'
   for i=2,#extensionList do
      findOptions = findOptions .. ' -o -iname "*.' .. extensionList[i] .. '"'
   end

   -- find the image path names
   self.imagePath = torch.CharTensor()  -- path to each image in dataset
   self.imageClass = torch.LongTensor() -- class index of each image (class index in self.classes)
   self.classList = {}                  -- index of imageList to each image of a particular class
   
--    print('running "find" on each class directory, and concatenate all'
--        .. ' those filenames into a single file containing all image paths for a given class')
   -- so, generates one file per class
   local classFindFiles = {}
   for i=1,#self.classes do
      classFindFiles[i] = os.tmpname()
   end
   local combinedFindList = os.tmpname();
   
   -- iterate over classes
   local tmpfile = os.tmpname()
   local tmphandle = assert(io.open(tmpfile, 'w'))
   -- iterate over classes
   for i, class in ipairs(self.classes) do
      -- iterate over classPaths
      for j,path in ipairs(classPaths[i]) do
         local command = find .. ' "' .. path .. '" ' .. findOptions 
            .. ' >>"' .. classFindFiles[i] .. '" \n'
         tmphandle:write(command)
      end
   end
   io.close(tmphandle)
   os.execute('bash ' .. tmpfile)
   os.execute('rm -f ' .. tmpfile)
   
--    print('Now combine all the files to a single large file')
   local tmpfile = os.tmpname()
   local tmphandle = assert(io.open(tmpfile, 'w'))
   -- concat all finds to a single large file in the order of self.classes
   for i=1,#self.classes do
      local command = 'cat "' .. classFindFiles[i] .. '" >>' .. combinedFindList .. ' \n'
      tmphandle:write(command)
   end
   io.close(tmphandle)
   os.execute('bash ' .. tmpfile)
   os.execute('rm -f ' .. tmpfile)
   
   --==========================================================================
--    print('Load the large concatenated list of sample paths to self.imagePath')
   local maxPathLength = tonumber(sys.fexecute(wc .. " -L '" 
                                                  .. combinedFindList .. "' |" 
                                                  .. cut .. " -f1 -d' '")) + 1
   local length = tonumber(sys.fexecute(wc .. " -l '" 
                                           .. combinedFindList .. "' |" 
                                           .. cut .. " -f1 -d' '"))
   assert(length > 0, "Could not find any image file in the given input path")
   assert(maxPathLength > 0, "paths of files are length 0?")
--    assert(length == self.labels:size(1), "Number of image files does not match number of labels")
   self.imagePath:resize(length, maxPathLength):fill(0)
   local s_data = self.imagePath:data()
   local count = 0
   for line in io.lines(combinedFindList) do
      ffi.copy(s_data, line)
      s_data = s_data + maxPathLength
      if self.verbose and count % 10000 == 0 then 
         xlua.progress(count, length) 
      end; 
      count = count + 1
   end

   self.numSamples = self.imagePath:size(1)
   if self.verbose then print(self.numSamples ..  ' samples found.') end
   --==========================================================================
--    print('Updating classList and imageClass appropriately')
   self.imageClass:resize(self.numSamples)
   local runningIndex = 0
   for i=1,#self.classes do
      if self.verbose then xlua.progress(i, #(self.classes)) end
      local length = tonumber(sys.fexecute(wc .. " -l '" 
                                              .. classFindFiles[i] .. "' |" 
                                              .. cut .. " -f1 -d' '"))
      if length == 0 then
         error('Class has zero samples')
      else
         self.classList[i] = torch.linspace(runningIndex + 1, runningIndex + length, length):long()
         self.classList[self.classes[i]] = self.classList[i]
         self.imageClass[{{runningIndex + 1, runningIndex + length}}]:fill(i)
      end
      runningIndex = runningIndex + length
   end

   --==========================================================================
   -- clean up temporary files
--    print('Cleaning up temporary files')
   local tmpfilelistall = ''
   for i=1,#(classFindFiles) do
      tmpfilelistall = tmpfilelistall .. ' "' .. classFindFiles[i] .. '"'
      if i % 1000 == 0 then
         os.execute('rm -f ' .. tmpfilelistall)
         tmpfilelistall = ''
      end
   end
   os.execute('rm -f '  .. tmpfilelistall)
   os.execute('rm -f "' .. combinedFindList .. '"')
   --==========================================================================
end

local function prepareForDoGet(self, modality, classes, i1, i2)

   local indices, quantity
   if type(i1) == 'number' then
      if type(i2) == 'number' then -- range of indices
         indices = torch.range(i1, i2); 
         quantity = i2 - i1 + 1;
      else -- single index 
         indices = {i1}; quantity = 1 
      end 
   elseif type(i1) == 'table' then
      indices = i1; quantity = #i1;         -- table
   elseif (type(i1) == 'userdata' and i1:nDimension() == 1) then
      indices = i1; quantity = (#i1)[1];    -- tensor
   else
      error('Unsupported input types: ' .. type(i1) .. ' ' .. type(i2))      
   end
   assert(quantity > 0)
   -- now that indices has been initialized, get the samples

   local data
   if modality == 'I' then
      self.bufferedData = torch.FloatTensor(quantity, self.sampleSize[1], self.sampleSize[2], self.sampleSize[3])
   else
      self.bufferedData = torch.FloatTensor(quantity, self.tags:size(2))
   end
   self.bufferedLabels = torch.FloatTensor(quantity, self.labels:size(2))

   local imgPathIndices
   if classes then
      imgPathIndices = torch.LongTensor()
      for c = 1,#classes do
         imgPathIndices = torch.cat(imgPathIndices, self.classList[classes[c]], 1)
      end
   end

   return indices, quantity, imgPathIndices, self.bufferedData, self.bufferedLabels
end

-- local function doGetImagesParallel(self, indices, quantity, imgPathIndices, data, labels)
local function doGetImagesParallel(self, classes, i1, i2, perm, async)

   local indices, quantity, imgPathIndices, data, labels = prepareForDoGet(self, 'I', classes, i1, i2)

   local self_labels = self.labels
   local self_imagePath = self.imagePath
   local ls = self.loadSize

--    local nThreads = 16
--    local pool = threads.Threads(nThreads)

   for i = 1,quantity do
      pool:addjob(function()
         
        local ffi = require 'ffi'
        local gm = require 'graphicsmagick'

        local idx = indices[i]
        if perm then
           idx = perm[idx]
        end
         
        local imgpath
        if classes then
            imgpath = ffi.string(torch.data(self_imagePath[imgPathIndices[idx]]))
        else
            imgpath = ffi.string(torch.data(self_imagePath[idx]))
        end

        local out = gm.Image()
        out:load(imgpath, ls[3], ls[2])
        out = out:toTensor('float','RGB','DHW') -- multiply by 255 if using matlab-generated channel avgs and stdevs

        data[i] = out
        local imNum = string.match(imgpath, '%d+')
        labels[i] = self_labels[imNum]
        if quantity == 1 then -- For debugging purposes
           print(string.format('i = %d, imgpath = %s\n', i, imgpath))
        end

      end)
   end

   if not async then
      pool:synchronize()
      return data, labels
   end
--    print(labels)
--    pool:terminate()
end

local function loadImage(imgpath, loadSize)
   local out = gm.Image()
   out:load(imgpath, loadSize[3], loadSize[2])
   --    :size(self.sampleSize[3], self.sampleSize[2])
   out = out:toTensor('float','RGB','DHW') -- multiply by 255 if using matlab-generated channel avgs and stdevs
   return out
end

local function doGetModality(self, modality, classes, i1, i2, perm)

   local indices, quantity, imgPathIndices, data, labels = prepareForDoGet(self, modality, classes, i1, i2)

   for i=1,quantity do

      local idx = indices[i]
      if perm then
         idx = perm[idx]
      end

      -- load the sample
      local imgpath 
      if classes then
         imgpath = ffi.string(torch.data(self.imagePath[imgPathIndices[idx]]))
      else
         imgpath = ffi.string(torch.data(self.imagePath[idx]))
      end

      local imNum = string.match(imgpath, '%d+')
      labels[i] = self.labels[imNum]
      if modality == 'I' then
         data[i] = loadImage(imgpath, self.loadSize)
      else
         data[i] = self.tags[imNum]
      end

      -- For debugging purposes only
      if quantity == 1 then
         print(imgpath)
      end
   end
   return data, labels
end

-- TODO: Remove?
-- converts a table of samples (and corresponding labels) to a clean tensor
local function tableToOutput(self, dataTable, labelTable)
   local data, labels
   local quantity = #labelTable
   if quantity == 1 then
      data = dataTable[1]
      labels = labelTable[1]
   else
      data = torch.FloatTensor(quantity, self.sampleSize[1], self.sampleSize[2], self.sampleSize[3])
      labels = torch.ByteTensor(quantity, 24) -- TODO: This is only for mirflickr
      for i=1,#dataTable do
         data[{ {i}, {}, {}, {} }]:copy(dataTable[i])
	     labels[{ {i}, {} }]:copy(labelTable[i])
      end
   end   
   return data, labels
end

function dataset:size(class, list)
   list = list or self.classList
   if not class then
      return self.numSamples
   elseif type(class) == 'string' then
      return list[self.classIndices[class]]:size(1)
   end
end

function dataset:sizeTrain()
   return self:size('training')
end

function dataset:sizeTest()
   return self:size('query')
end

function dataset:sizeVal()
   return self:size('val')
end

function dataset:sizePretraining()
   return self:size('pretraining')
end

function dataset:normalize(data)

   for i=1,3 do
      data[{ {}, {i}, {}, {} }]:add(-self.channelAvgs[i])
      data[{ {}, {i}, {}, {} }]:div(self.channelStds[i])
   end
   return data
end

-- TODO: Remove
-- by default, just load the image and return it
function dataset:defaultSampleHook(imgpath)
--    ldTimer:resume()
   local out = gm.Image()
   out:load(imgpath, self.loadSize[3], self.loadSize[2])
   :size(self.sampleSize[3], self.sampleSize[2])
--    out:load('/home/kejosl/Datasets/mirflickr/ImageData/training/im14261.jpg', self.loadSize[3], self.loadSize[2])
--    out:load('/home/kejosl/Datasets/mirflickr/ImageData/training/im14261.jpg', 500, 349)
   out = out:toTensor('float','RGB','DHW') -- multiply by 255 if using matlab-generated channel avgs and stdevs
--    ldTimer:stop()
   -- Normalization
--    nmTimer:resume()
--    for i=1,3 do
--       out[{ {i}, {}, {} }]:add(-self.channelAvgs[i])
--       out[{ {i}, {}, {} }]:div(self.channelStds[i])
--    end
--    nmTimer:stop()
   return out
end

-- TODO: fix or remove
-- sampler, samples from the training set.
function dataset:sample(quantity)
   quantity = quantity or 1
   local dataTable = {}
   local scalarTable = {}   
   for i=1,quantity do
      local class = torch.random(1, #self.classes)
      local out = self:getByClass(class)
      table.insert(dataTable, out)
      table.insert(scalarTable, class)      
   end
   local data, scalarLabels, labels = tableToOutput(self, dataTable, scalarTable)
   return data, scalarLabels, labels      
end

function dataset:getImagePath(index)
   return ffi.string(torch.data(self.imagePath[index]))
end

function dataset:getImagePathByClass(class, index)
   return ffi.string(torch.data(self.imagePath[self.classList[class][index]]))
end

function dataset:loadImageBatchBySplitAsync(classArg, modality, i1, i2, permutation)
    -- 'Class' means the same thing as 'split' in this sense.
    -- classArg can be a string or a table of strings, belonging to 'training', 'query', 'val', or 'pretraining'.

   local classes
   if type(classArg) == 'string' then
      classes = {classArg}
   elseif type(classArg) == 'table' then
      classes = classArg
   end

   local quantity
   if not i2 then 
      quantity = 1
   else
      quantity = i2 - i1 + 1
   end

   doGetImagesParallel(self, classes, i1, i2, permutation, true)
end

function dataset:getBufferedBatch()

    pool:synchronize()
    self.bufferedData = self:normalize(self.bufferedData)
    return self.bufferedData, self.bufferedLabels
end

function dataset:getBySplit(classArg, modality, i1, i2, permutation)
    -- 'Class' means the same thing as 'split' in this sense.
    -- classArg can be a string or a table of strings, belonging to 'training', 'query', 'val', or 'pretraining'.

   local classes
   if type(classArg) == 'string' then
      classes = {classArg}
   elseif type(classArg) == 'table' then
      classes = classArg
   end

   local quantity
   if not i2 then 
      quantity = 1
   else
      quantity = i2 - i1 + 1
   end

   local data, labels

   if modality == 'I' then
      if quantity > 8 then
         data, labels = doGetImagesParallel(self, classes, i1, i2, permutation)
      else
         data, labels = doGetModality(self, 'I', classes, i1, i2, permutation)
      end
      data = self:normalize(data)
   elseif modality == 'X' then
      data, labels = doGetModality(self, 'X', classes, i1, i2, permutation)
   else
      print('Error in getBySplit: unrecognized modality')
   end

   return data, labels
end