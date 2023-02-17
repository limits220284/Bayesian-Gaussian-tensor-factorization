clear;
u=[1,2;3,4];
v=[1,3;2,4;5,6;];
x=[1,5;2,6;3,7;4,8;];
MatHat = zeros(length(u),length(v),length(x));
dim = size(MatHat);
for i = 1:dim(1)
    for j = 1:dim(2)
        for t = 1:dim(3)
            MatHat(i,j,t) = u(i,:) * (v(j,:) .* x(t,:))';
        end
    end
end