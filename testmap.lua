
mapData = torch.load('mapData.t7')
queryCodes = mapData.queryCodes
databaseCodes = mapData.databaseCodes
queryLabels = mapData.queryLabels
databaseLabels = mapData.databaseLabels

Q = queryCodes:size(1)
sumAPs = 0
for q = 1,Q do
    query = torch.repeatTensor(queryCodes[q], databaseCodes:size(1), 1, 1)
    ne = torch.ne(query, databaseCodes):sum(2)
    ne = torch.reshape(ne, ne:size(1))
    topkResults, ind = torch.Tensor(ne:size(1)):copy(ne):topk(K)

    topkResults_sorted, ind2 = torch.sort(topkResults)
    topkIndices = ind:index(1,ind2)

    qLabel = queryLabels[q]

    AP = 0
    correct = 0
    for k = 1,K do

        kLabel = databaseLabels[topkIndices[k]]
        dotProd = torch.dot(qLabel, kLabel)
        if dotProd > 0 then
            correct = correct + 1
            AP = AP + (correct / k) -- add precision component
        end
    end
    if correct > 0 then -- Correct should only be 0 if there are a small # of database objects and/or poorly trained
        AP = AP / correct -- Recall component (divide by number of ground truth positives in top k)
    end
    sumAPs = sumAPs + AP
end
mAP = sumAPs / Q