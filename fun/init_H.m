function H = init_H(X, c, n)
index = runFastMICE(X, c);
H=zeros(n,c);
for i=1:n
    indexs=index(i,1);
    H(i,indexs)=1;
end
H=H';
end