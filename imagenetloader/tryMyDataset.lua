require 'dataset'
local ffi = require 'ffi'

-- dataset = imageLoader{path='/home/kjoslyn/datasets/mirflickr', split=100, sampleSize={3,227,227}, splitFolders={'training'}}
dataset = imageLoader{path='/home/kejosl/Datasets/mirflickr/ImageData', split=100, sampleSize={3,227,227}, splitFolders={'training', 'val'}}

-- local inputs, labels = dataset:sample(1)