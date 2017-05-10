function [ output_args ] = plotResults( evalInterval, numEpochs, trainOrValResMat, title, legendCell )

    load('colorOrder10.mat');
    set(gca, 'ColorOrder', colorOrder, 'NextPlot', 'replacechildren');
    plot(evalInterval:evalInterval:numEpochs, trainOrValResMat.');
    legend(legendCell)
    title(title)

end

