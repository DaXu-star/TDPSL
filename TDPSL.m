%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                       %
% This is the source code for the consensus funtion in the FastMICE     %
% algorithm. If you find it helpful in your research, please cite the   %
% paper below.                                                          %
%                                                                       %
% Da Xu, Jialin Zhang, Guoliang Chen£¬Guoxian Yu                        %
% Tensor-driven Denoising Projection Self-representation Learning for   %
% Multi-view Unsupervised Feature Selection                             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ranking] = TDPSL(X,alpha,beta,deta,gamma,c,omega)
V = size(X,2); 
n = size(X{1,1},1); 
%% Initialize similarity matrix Zv
knn = 10;
for i = 1:V
    Zv{i} = constructW_PKN(X{i}',knn);
end
%% Initialize P, D and B
P = cell(1,V);
m = cell(1,V);
D = cell(1,V);
H = init_H(X, c, n);
for i=1:V
    m{1,i} = size(X{1,i},2); %Feature dimension
    X{1,i} = X{1,i}'; %Each column of X_v represents a sample
    P{1,i} = eye(m{1,i},c);
    D{1,i} = eye(m{1,i});
    Y{i} = zeros(size(X{1,i},1),n);
    Ev{i} = zeros(size(X{1,i},1),n);
    G{i} = zeros(n,n);
    W{i} = zeros(n,n);  %multiplier
end
%% initialize 
pho = 0.1;
mu = 0.1;
pho_max = 10e6;
mu_max = 10e6;
eta1 = 2;
eta2 = 2;
sX = [n, n, V];
%% Optimization
MAXITER = 20; %Maximum number of iterations
for iter = 1:MAXITER
    %% Update P, D
    for i = 1:V
        temp_W = ((X{1,i}*Zv{1,i}) * (X{1,i}*Zv{1,i})') + deta * D{1,i};
        P{1,i} = temp_W \ (X{1,i}*Zv{1,i}* H');
        tempD = 0.5 * (sqrt(sum(P{1,i}.^2,2) + eps)).^(-1);
        D{1,i} = diag(tempD);
    end
    
    %% Update Q
    Q = max(H,0);
    
    %% Update H
    SVD = zeros(size(H));
    for i = 1:V
        SVD = SVD + (P{1,i})'*X{1,i}*Zv{1,i};
    end
    SVD = SVD + gamma * Q;
    [V_H,~,U_H] = svd(SVD,'econ');
    H = V_H * U_H';
    
    %% Update Ev
    for i=1:V
        F1 = X{1,i}-X{1,i}*Zv{i}+Y{i}./mu;
        Ev{i} = solve_l1l2(F1,beta./mu);
    end
    
    %% Update Zv
    for i = 1:V
        B1 = X{1,i}-Ev{i}+Y{i}./mu;
        B2 = G{i}-W{i}./pho;
        Zv{i} = inv(2*X{1,i}'*P{1,i}*P{1,i}'*X{1,i}+mu*X{1,i}'*X{1,i}+pho*eye(n,n))*(2*X{1,i}'*P{1,i}*H+mu*X{1,i}'*B1+pho*B2);
    end
    
    %% update G{i}
    Z_tensor = cat(3, Zv{:,:});
    W_tensor = cat(3, W{:,:});
    z = Z_tensor(:);
    w = W_tensor(:);
    [g, ~] = wshrinkObj(z+1/pho*w,alpha/pho,sX,0,3,omega);
    G_tensor = reshape(g, sX);
    for i=1:V
        G{i} = G_tensor(:,:,i);
    end
    
    % == update W{i} ==
    for i=1:V
        W{i} = W{i}+pho*(Zv{i}-G{i});
    end
    
    % == update Y{i} ==
    for i=1:V
        Y{i} = Y{i}+mu*(X{1,i}-X{1,i}*Zv{i}-Ev{i});
    end
    
    % == update pho  mu ==
    pho = min(pho*eta1,pho_max);
    mu  = min(mu*eta2,mu_max);
end

%Calculate feature ranking
WW = [];
for i = 1:V
    WW = [WW;P{1,i}];
end
[~,ranking] = sort(sum(WW.*WW,2),'descend');
end

