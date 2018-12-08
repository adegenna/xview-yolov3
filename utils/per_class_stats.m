function [class_mu, class_sigma, class_cov] = per_class_stats(classes,w,h)
% measure the min and max bbox sizes for rejecting bad predictions
area = log(w.*h);
aspect_ratio = log(w./h);
uc = unique(classes(:)); 
n = numel(uc);

class_mu = zeros(n,4,'single'); % width, height, area, aspect_ratio
class_sigma = zeros(n,4,'single'); 
class_cov = zeros(n,4,4,'single');

for i = 1:n
    j = find(classes==uc(i));
    wj = log(w(j));  hj = log(h(j));  aj = area(j);  arj = aspect_ratio(j,:);
    data = [wj, hj, aj, arj];
    class_mu(i,:) = mean(data,1);
    class_sigma(i,:) = std(data,1);
    class_cov(i,:,:) = cov(data);
    %[~,C] = kmeans([wj hj],1,'MaxIter',5000,'OnlinePhase','on');
    
    %class_name = xview_names(xview_classes2indices(uc(i)));
    %close all; hist211(arj,hj,40); title(sprintf('%s, cor = %g',class_name,corr(wj,hj)))
    
    % close all; hist211(wj,hj,{linspace(0,max(wj),40),linspace(0,max(hj),40)}); plot(C(:,1),C(:,2),'g.','MarkerSize',50); title(sprintf('%s, cor = %g',class_name,corr(wj,hj)))
    % ha=fig; histogram(arj, linspace(-3,3,50)); title(class_name); ha.YScale='linear';
end
end
