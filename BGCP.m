clear;
load tensor.mat;
%张量大小为214 * 61 *144
% OriMat(:,:,1) = [31,42,65;63,86,135;];
% OriMat(:,:,2) = [38,52,82;78,108,174;];
% OriMat(:,:,3) = [45,62,99;93,130,213;];
% OriMat(:,:,4) = [52,72,116;108,152,252;];
OriMat = tensor;
dim = size(OriMat);

RandMat = rand(dim);
RandMat0 = RandMat + 0.5 - 0.2;
SparseMat = round(RandMat0).*OriMat;
pos1 = find(SparseMat>0);
pos2 = find(OriMat>0 & SparseMat==0);
BinMat = SparseMat;
BinMat(pos1) = 1;

%设置初始化参数大小
r = 40;
maxiter = 200;
FactMat = cell(3,1);
for k=1:3
    FactMat{k} = 0.1*randn(dim(k),r);
end
tau = 1;
a0 = 1e-6;
b0 = 1e-6;
lambda = ones(r,1);
c0 = 1e-6;
h0 = 1e-6;

%根据前面已经推导的后验分布进行迭代采样
%第一步:对因子矩阵U,V,X进行采样更新
rmse = zeros(maxiter,1);
for iter = 1:maxiter
    %1、更新因子矩阵U的每一列ui
    for i = 1:dim(1)
        tot_lambda = zeros(r,r);
        tot_mu = zeros(r,1);
        for j = 1:dim(2)
            for t = 1:dim(3)
                if SparseMat(i,j,t) ~= 0
                    wjt = FactMat{2}(j,:) .* FactMat{3}(t,:);
                    tot_lambda = tot_lambda + wjt' * wjt; 
                    tot_mu = tot_mu + SparseMat(i,j,t) * wjt';
                end
            end
        end
        Lambda = (diag(lambda) + tau * tot_lambda)^(-1);
        Lambda = (Lambda + Lambda')./2;
        mu = tau * Lambda * tot_mu;
        FactMat{1}(i,:) = mvnrnd(mu,Lambda);%mvnrnd作用是生成len(mu)个协方差为Lambda的数据作为u_i的后验分布
    end
    %2、更新因子矩阵V的每一列
    for j = 1:dim(2)
        tot_lambda = zeros(r,r);
        tot_mu = zeros(r,1);
        for i = 1:dim(1)
            for t = 1:dim(3)
                if SparseMat(i,j,t) ~= 0
                    wit = FactMat{1}(i,:) .* FactMat{3}(t,:);
                    tot_lambda = tot_lambda + wit' * wit; 
                    tot_mu = tot_mu + SparseMat(i,j,t) .* wit';
                end
            end
        end
        Lambda = (diag(lambda) + tau .* tot_lambda)^(-1);
        Lambda = (Lambda + Lambda')./2;
        mu = tau .* Lambda * tot_mu;
        FactMat{2}(j,:) = mvnrnd(mu,Lambda);%mvnrnd作用是生成len(mu)个协方差为Lambda的数据作为v_j的后验分布
    end
    %3、更新因子矩阵X的每一列
    for t = 1:dim(3)
        tot_lambda = zeros(r,r);
        tot_mu = zeros(r,1);
        for i = 1:dim(1)
            for j = 1:dim(2)
                if SparseMat(i,j,t) ~= 0
                    wij = FactMat{1}(i,:) .* FactMat{2}(j,:);
                    tot_lambda = tot_lambda + wij' * wij; 
                    tot_mu = tot_mu + SparseMat(i,j,t) .* wij';
                end
            end
        end
        Lambda = (diag(lambda) + tau .* tot_lambda)^(-1);
        Lambda = (Lambda + Lambda')./2;
        mu = tau .* Lambda * tot_mu;
        FactMat{3}(t,:) = mvnrnd(mu,Lambda);%mvnrnd作用是生成len(mu)个协方差为Lambda的数据作为x_t的后验分布
    end
    %得到原张量
    MatHat = OriMat;
    for i = 1:dim(1)
        for j = 1:dim(2)
            for t = 1:dim(3)
                MatHat(i,j,t) = FactMat{1}(i,:) * (FactMat{2}(j,:) .* FactMat{3}(t,:))';
            end
        end
    end
    rmse(iter,1) = sqrt(sum((OriMat(pos2)-MatHat(pos2)).^2)./length(pos2));
    %4、更新超参数\lambda
    c = (0.5*sum(dim)+c0) * ones(r,1);
    h = 0;
    for k = 1:3
        h = h + diag(FactMat{k}'*FactMat{k});
    end
    h = h0 + 0.5 * h;
    lambda = gamrnd(c,1./h);%为什么要加一个1./h?

    %5、更新超参数\tau
    a = a0 + 0.5 * length(pos1);
    error = SparseMat - MatHat;
    b = b0 + 0.5 * sum(error(pos1).^2);
    tau = gamrnd(a,1./b);

    %6、输出迭代误差
    fprintf('itertion = %g, RMSE = %g km/h.\n',iter,rmse(iter));
end



