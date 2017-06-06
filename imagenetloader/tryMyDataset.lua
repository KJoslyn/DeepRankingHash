require 'dataset'
local ffi = require 'ffi'

function initDataset(splitFolders)
    dataset = imageLoader{path='/home/kejosl/Datasets/mirflickr/ImageData', sampleSize={3,227,227}, splitFolders=splitFolders}
end

initDataset({'training', 'query'})
da, la = dataset:getBySplit({'training', 'query'},1,128);

-- dataset = imageLoader{path='/home/kjoslyn/datasets/mirflickr', sampleSize={3,227,227}, splitFolders={'training'}}