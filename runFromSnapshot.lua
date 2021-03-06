require 'fullNet'
require 'main'

-- TODO: Check all of these
local numRuns = 1 -- Number of runs for each parameter setting
local datasetType = 'nus'
local numEpochs = 300
local evalInterval = 10
local consecutiveStop = 10
local statsId = 29
local startEpoch = 161
loadParamsAndPackages(datasetType, 50, false)

local autoStatsDir
if datasetType == 'mir' then
    autoStatsDir = 'mirflickr'
elseif datasetType == 'nus' then
    autoStatsDir = 'nuswide'
end
autoStatsDir = autoStatsDir .. '/CM'

local tempPath = g.userPath .. '/kevin/Project/temp/'

local gSaved = torch.load(tempPath .. 'gSaved.t7') -- TODO: Create empty

g.statsDir = g.userPath .. '/kevin/Project/autoStats/' .. autoStatsDir
g.meta = io.open(g.statsDir .. "/metaStats.txt", 'a')

p.numEpochs = numEpochs
p.evalInterval = evalInterval
p.consecutiveStop = consecutiveStop
g.startStatsId = statsId
p.numRuns = numRuns

pfsS = torch.load(tempPath .. 'pfsS.t7') -- TODO: Create
local numParamCombs = #pfsS/numRuns -- This is only used before a non-nil gSaved is loaded

g.paramSettingsLegend = gSaved.paramSettingsLegend or {}
g.trainResultsMatrix = gSaved.trainResultsMatrix or torch.Tensor(numParamCombs, numEpochs / evalInterval):fill(0)
g.valResultsMatrix = gSaved.valResultsMatrix or torch.Tensor(numParamCombs, numEpochs / evalInterval):fill(0)
g.resultsParamIdx = gSaved.resultsParamIdx or 0
g.endStatsId = gSaved.endStatsId or nil

pfs = pfsS[#pfsS]

for _, value in pairs(pfs) do
    paramName = value[1]
    paramValue = value[2][1]
    print("name")
    print(paramName)
    print("value")
    print(paramValue)
    setParamValue(paramName, paramValue)
end

-- We are working backwards from the end so p.run == numRuns signals start of new parameter setting
if p.run == numRuns then
    g.resultsParamIdx = g.resultsParamIdx + 1
end
runWithParams(pfs)

pfsS[#pfsS] = nil
torch.save(tempPath .. 'pfsS.t7', pfsS)

gSaved = {}
gSaved.paramSettingsLegend = g.paramSettingsLegend
gSaved.trainResultsMatrix = g.trainResultsMatrix
gSaved.valResultsMatrix = g.valResultsMatrix
gSaved.resultsParamIdx = g.resultsParamIdx
gSaved.endStatsId = g.endStatsId
torch.save(tempPath .. 'gSaved.t7', gSaved)

io.close(g.meta)
