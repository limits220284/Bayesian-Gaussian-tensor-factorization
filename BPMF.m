clear;
load tensor.mat;
%将该张量展开成大小为214 \times 8784大小的矩阵
OriMat = ten2mat(tensor,[214,61,144],1); % original matrix

% 紧接着，生成一个0-1之间均匀分布的随机矩阵
% 调整该随机矩阵元素数值的大小可用以控制缺失率
% 以下设置的缺失率为20%。

dim = size(OriMat);
RandMat = rand(dim); % uniform distributed random numbers
RandMat0 = RandMat+0.5-0.2;
SparseMat = round(RandMat0).*OriMat; % sparse matrix with 20% missing ratio
pos1 = find(SparseMat>0);%剩余矩阵的索引,按列排列成列向量
pos2 = find(OriMat>0 & SparseMat==0);%原来矩阵中大于零的,并且生成的稀疏矩阵中等于零的
BinMat = SparseMat;
BinMat(pos1) = 1; % binary matrix

%设置初始化参数的大小。
r = 40;
maxiter = 200;
FactMat = cell(2,1);
for k = 1:2
    FactMat{k} = 0.1*randn(dim(k),r);
end
tau = 1;
a0 = 1e-6;
b0 = 1e-6;
lambda = ones(r,1);
c0 = 1e-6;
h0 = 1e-6;

% 根据前面已经推导的后验分布进行迭代采样。
% 第一步:对因子矩阵U,V进行采样更新。
rmse = zeros(maxiter,1);
for iter = 1:maxiter
    % Update factor matrices
    for i = 1:dim(1)
        pos = find(SparseMat(i,:)>0);%第u_i行数值不为零的数的索引
        var = FactMat{2}(pos,:);%对应的v_j
        Lambda = (tau*var'*var+diag(lambda))^(-1);
        Lambda = (Lambda+Lambda')./2;%保证协方差矩阵是对称矩阵
        mu = tau*Lambda*var'*SparseMat(i,pos)';
        FactMat{1}(i,:) = mvnrnd(mu,Lambda);%mvnrnd作用是生成len(mu)个协方差为Lambda的数据作为u_i的后验分布
    end
    for i = 1:dim(2)
        pos = find(SparseMat(:,i)>0);
        var = FactMat{1}(pos,:);
        Lambda = (tau*var'*var+diag(lambda))^(-1);
        Lambda = (Lambda+Lambda')./2;
        mu = tau*Lambda*var'*SparseMat(pos,i);
        FactMat{2}(i,:) = mvnrnd(mu,Lambda);
    end
    MatHat = OriMat;
    for i = 1:dim(1)
        for j = 1:dim(2)
            MatHat(i,j) = FactMat{1}(i,:) * FactMat{2}(j,:)';
        end
    end
    rmse(iter,1) = sqrt(sum((OriMat(pos2)-MatHat(pos2)).^2)./length(pos2));

% 第二步:对超参数\lambda进行采样更新。
% Update lambda
    c = (0.5*sum(dim)+c0)*ones(r,1);
    h = 0;
    for k = 1:2
        h = h+diag(FactMat{k}'*FactMat{k});
    end
    h = h0+0.5*h;
    lambda = gamrnd(c,1./h);%为什么要加一个1./h?
% 第三步:对精度项\tau进行采样更新，同时输出当前迭代的误差。
    % Update precision
    a = a0+0.5*length(pos1);
    error = SparseMat-MatHat;
    b = b0+0.5*sum(error(pos1).^2);
    tau = gamrnd(a,1./b);
    
    % Print the results
    fprintf('iteration = %g, RMSE = %g km/h.\n',iter,rmse(iter));
end
