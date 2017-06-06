
function getAvgNumSimilarInDB(classesFrom, classesTo)

    -- We only care about the labels, not the codes
    local qc, ql = getCodesAndLabels(X, classesFrom, true)
    local dc, dl = getCodesAndLabels(X, classesTo, true)

    local count = 0
    for i = 1, ql:size(1) do
        local lq = ql[i]
        for j = 1,dl:size(1) do
            local ld = dl[j]
            local d = torch.dot(lq, ld)
            if d > 0 then
                count = count + 1
            end
        end
    end

    print(count)

    return count / ql:size(1)
end