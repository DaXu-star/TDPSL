function Label = runFastMICE(fea, trueK, Msize, p, Knn, distance)
if nargin < 3
    distance = 'euclidean';
end
if nargin < 3
    Knn = 5;
end
if nargin < 3
    p = 1000;
end
if nargin < 3
    Msize = 20;
end


lowerFeatureRatio = 0.2;   % Minimum ratio of feature sampling
upperFeatureRatio = 0.8;   % Maximum ratio of feature sampling

% Msize = 20; % Number of base clusterings
% p = 1000;   % Total number of anchors in a view group
% Knn = 5;    % Total number of nearest neighbors in a view group
% For text dataset, the Cosine distance is suggested.
% distance = 'cosine'; 

lowerK = 1; % The minimum number of clusters in a base clustering is lowerK * trueK
upperK = 2; % Maximum number of clusters in a base clustering

nView = length(fea);
N = size(fea{1},1);

% The number of anchors cannot exceed the number of instances.
if p > N
    p = N; 
end
    
%% Generate base clusterins
minK = lowerK*trueK;
maxK = upperK*trueK;
IDX = FastMICE_EnsembleGeneration(fea, Msize, p, Knn, minK, maxK,lowerFeatureRatio,upperFeatureRatio,distance);
Label = FastMICE_ConsensusFunction(IDX,trueK,20,5);
end
     
