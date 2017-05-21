require 'torch'
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
          avgStdFileFound = true
        --   if string.match(dirpath, 'avgStdTrainSet.mat') then
        --      avgStdFileFound = true
        --   end
       end
          
       return labelFileFound and avgStdFileFound
   end,
    name="path",
    type="string",
    help="Path of directory with images, containing folders \'training\', \'val\', \'query\', \'pretraining\', and avgStdTrainSet.mat and labels.mat"}, 

   {name="sampleSize",
    type="table",
    help="a consistent sample size to resize the images"},

   {name="splitFolders",
    type="table",
    help="examples: training, val, query, pretraining"},

   {name="split",
    type="number",
    help="Percentage of split to go to Training",
    default = 90},

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

   -- Load labels and channel avgs and stdevs
   if not matio then
      matio = require 'matio'
   end
   local labels = matio.load(self.path .. '/labels.mat')
   self.labels = labels.labels
--    local avgStd = matio.load(path .. '/avgStdevTrainSet.mat')
--    avgStd = avgStd.avgStd
--    self.channelAvgs = avgStd.means:reshape(3)
--    self.channelStds = avgStd.stdev:reshape(3)

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
   
   print('running "find" on each class directory, and concatenate all' 
         .. ' those filenames into a single file containing all image paths for a given class')
   -- so, generates one file per class
   local classFindFiles = {}
   for i=1,#self.classes do
      classFindFiles[i] = os.tmpname()
   end
   local combinedFindList = os.tmpname();
   
   local tmpfile = os.tmpname()
   local tmphandle = assert(io.open(tmpfile, 'w'))
   -- iterate over classes
   print("--Classes")
   print(self.classes)
   print("--Classpaths")
   print(classPaths)
   print("--classFindFiles")
   print(classFindFiles)
   print("--find")
   print(find)
   print("--findOptions")
   print(findOptions)

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
   
   print('now combine all the files to a single large file')
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
   print('load the large concatenated list of sample paths to self.imagePath')
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
   print('Updating classList and imageClass appropriately')
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
         self.imageClass[{{runningIndex + 1, runningIndex + length}}]:fill(i)
      end
      runningIndex = runningIndex + length
   end

   --==========================================================================
   -- clean up temporary files
   print('Cleaning up temporary files')
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

local function doGet(self, class, i1, i2)

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

   local data = torch.FloatTensor(quantity, self.sampleSize[1], self.sampleSize[2], self.sampleSize[3])
   local labels = torch.ByteTensor(quantity, 24)
   for i=1,quantity do
      -- load the sample
      local imgpath 
      if class then
         imgpath = ffi.string(torch.data(self.imagePath[self.classList[class][indices[i]]]))
      else
         imgpath = ffi.string(torch.data(self.imagePath[indices[i]]))
      end
      data[i] = self:sampleHookTest(imgpath)

      local imNum = string.match(imgpath, '%d+')
      labels[i] = self.labels[imNum]

      if quantity == 1 then
         print(imgpath)
      end
   end

--    local dataTable = {}
--    local labelTable = {}
--    for i=1,quantity do
--       -- load the sample
--       local imgpath = ffi.string(torch.data(self.imagePath[indices[i]]))
--       out = self:sampleHookTest(imgpath)
--       table.insert(dataTable, out)

--       local imNum = string.match(imgpath, '%d+')
--       table.insert(labelTable, self.labels[imNum])      
--    end
--    local data, labels = tableToOutput(self, dataTable, labelTable)
   return data, labels
end

-- converts a table of samples (and corresponding labels) to a clean tensor
local function tableToOutput(self, dataTable, labelTable)
   local data, labels
   local quantity = #labelTable
   if quantity == 1 then
      data = dataTable[1]
      labels = labelTable[1]
   else
      data = torch.FloatTensor(quantity, self.sampleSize[1], self.sampleSize[2], self.sampleSize[3])
      labels = torch.ByteTensor(quantity, 24)
      for i=1,#dataTable do
         data[{ {i}, {}, {}, {} }]:copy(dataTable[i])
	     labels[{ {i}, {} }]:copy(labelTable[i])
      end
   end   
   return data, labels
end

-- size(), size(class)
function dataset:size(class, list)
   list = list or self.classList
   if not class then
      return self.numSamples
   elseif type(class) == 'string' then
      return list[self.classIndices[class]]:size(1)
   end
end

-- size(), size(class)
function dataset:sizeTrain()
   return self:size('training')
end

-- size(), size(class)
function dataset:sizeTest()
   return self:size('query')
end

function dataset:sizeVal()
   return self:size('val')
end

function dataset:sizePretraining()
   return self:size('pretraining')
end

-- by default, just load the image and return it
function dataset:defaultSampleHook(imgpath)
   local out = gm.Image()
   out:load(imgpath, self.loadSize[3], self.loadSize[2])
   :size(self.sampleSize[3], self.sampleSize[2])
   out = out:toTensor('float','RGB','DHW') -- multiply by 255 if using matlab-generated channel avgs and stdevs
--    -- Normalization
--    for i=1,3 do
--       out[{ {i}, {}, {} }]:add(-self.channelAvgs[i])
--       out[{ {i}, {}, {} }]:div(self.channelStds[i])
--    end
   return out
end

-- TODO: fix or remove
-- sampler, samples from the training set.
function dataset:sample(quantity)
   if self.split == 0 then 
      error('No training mode when split is set to 0') 
   end
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

function dataset:get(i1, i2)
   return doGet(self, nil, i1, i2)
end

-- getByClass
function dataset:getByClass(class, i1, i2)
   return doGet(self, class, i1, i2)
end

function dataset:getByClasses(classes, i1, i2)
   return doGet(self, classes, i1, i2)
end
