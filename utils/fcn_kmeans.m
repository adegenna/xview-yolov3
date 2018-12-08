function C = fcn_kmeans(X, n)
rand("seed",0); randn("seed",0); %rng('default'); % For reproducibility
%X = [randn(100,2)*0.75+ones(100,2);
%    randn(100,2)*0.55-ones(100,2)];

% opts = statset('Display','iter');
%[idx,C, sumd] = kmedoids(X,n,'Distance','cityblock','Options',opts);

X = X(all(X<1000,2),:);

X = [X; X(:,[2, 1])];
[idx,C, sumd] = kmeans(X,n,'MaxIter',5000,'OnlinePhase','on');
%sumd


ha=figure();
for i = 1:numel(unique(idx))
    plot(X(idx==i,1),X(idx==i,2),'.','MarkerSize',4)
end

plot(C(:,1),C(:,2),'k.','MarkerSize',15)
title('Cluster Assignments and Means');
xlim=[0 1000]; ylim=[0 1000];
end

