function [ output_args ] = plotResults( evalInterval, numEpochs, trainOrValResMat, plotTitle, legendCell )

    load('colorOrder10.mat');
    set(gca, 'ColorOrder', colorOrder, 'NextPlot', 'replacechildren');
    plot(evalInterval:evalInterval:numEpochs, trainOrValResMat.');
    legend(legendCell)
    title(plotTitle)
    
    if size(trainOrValResMat,1) > 10
        axesHandle = gca;
        plotHandle = findobj(axesHandle, 'Type', 'line');
        for i = 11:size(trainOrValResMat,1)
            set(plotHandle(i), 'LineStyle', '--');
        end        
    end
end

