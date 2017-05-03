function calcMAPTest(fromModality, toModality, printToFile) -- TODO: Remove 3rd and 4th parameters

    fullModel:evaluate()

    if printToFile then
        date = os.date("*t", os.time())
        dateStr = date.month .. "_" .. date.day .. "_" .. date.hour .. "_" .. date.min
        outputFile = io.open(snapshotDir .. "/exampleRetrievals_" .. dateStr .. ".txt", "w")
    end

    K = 50

    queryCodes, databaseCodes, queryLabels, databaseLabels = getQueryAndDBCodesTest(fromModality, toModality)

    -- Q = 2
    Q = queryCodes:size(1)
    sumAPs = 0
    for q = 1,Q do

        if printToFile then
            outputFile:write("Query " .. q .. "\n")
        end

        -- databaseCodes = torch.reshape(databaseCodes, 4000, L)
        if fromModality == X then
            query = torch.repeatTensor(queryCodes[q], databaseCodes:size(1), 1, 1)
        else
            query = torch.repeatTensor(queryCodes[q], databaseCodes:size(1), 1)
        end

        ne = torch.ne(query, databaseCodes):sum(2)
        ne = torch.reshape(ne, ne:size(1))
        topkResults, ind = torch.Tensor(ne:size(1)):copy(ne):topk(K)

        ind2 = randSort(topkResults)
        -- topkResults_sorted, ind2 = torch.sort(topkResults)
        topkIndices = ind:index(1,ind2)

        qLabel = queryLabels[q]

        AP = 0
        correct = 0
        for k = 1,K do

            if printToFile and k <= 10 then
                outputFile:write(topkIndices[k] .. "\n")
            end

            kLabel = databaseLabels[topkIndices[k]]
            dotProd = torch.dot(qLabel, kLabel)
            if dotProd > 0 then
                correct = correct + 1
                AP = AP + (correct / k) -- add precision component
            end
            if k == 10 and printToFile then
                outputFile:write(string.format("Correct / 10 = %d\n", correct))
            end
        end
        if correct > 0 then -- Correct should only be 0 if there are a small # of database objects and/or poorly trained
            AP = AP / correct -- Recall component (divide by number of ground truth positives in top k)
        end
        if printToFile then
            outputFile:write(string.format("Correct / %d = %d\n", K, correct))
            outputFile:write(string.format("AP = %.2f\n", AP))
        end
        sumAPs = sumAPs + AP
    end
    mAP = sumAPs / Q

    if printToFile then
        outputFile:write(string.format("MAP = %.2f\n", mAP))
        io.close(outputFile)
    end

    return mAP
end

function calcMAP_old(queryCodes, databaseCodes, queryLabels, databaseLabels)

    K = 50

    -- Q = 1
    Q = queryCodes:size(1)
    sumAPs = 0
    for q = 1,Q do
        -- databaseCodes = torch.reshape(databaseCodes, 4000, L)
        if fromModality == X then
            query = torch.repeatTensor(queryCodes[q], databaseCodes:size(1), 1, 1)
        else
            query = torch.repeatTensor(queryCodes[q], databaseCodes:size(1), 1)
        end

        ne = torch.ne(query, databaseCodes):sum(2)
        ne = torch.reshape(ne, ne:size(1))
        topkResults, ind = torch.Tensor(ne:size(1)):copy(ne):topk(K)

        ind2 = randSort(topkResults)
        -- topkResults_sorted, ind2 = torch.sort(topkResults)
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

    return mAP
end