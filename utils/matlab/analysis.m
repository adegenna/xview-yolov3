% MATLAB xView analysis code for cleaning up targets and building a targets.mat file used during training

pkg load statistics;
pkg load symbolic;
clc; clear; close all
load xview_geojson.mat

% make_small_chips()
% return

chip_id = zeros(numel(chips),1);  % 1-847
chip_number = zeros(numel(chips),1);  % 5-2619
for i=1:numel(chips)
    chip_id(i) = find(strcmp(chips(i),uchips));
    
    s = chips{i};
    s = s(1:end-4);
    chip_number(i) = str2double(s);
end

uchips_numeric = zeros(numel(uchips),1);
for i=1:numel(uchips)
    uchips_numeric(i) = eval(uchips{i}(1:end-4));
end

% clean coordinates that fall off images (remove or crop)
image_h = shapes(chip_id,1);
image_w = shapes(chip_id,2);
[coords, v] = clean_coords(chip_number, coords, classes, image_h, image_w);
mean(v)

chip_id = chip_id(v);
chips = chips(v);
classes = classes(v);
image_h = image_h(v);
image_w = image_w(v); 
chip_number = chip_number(v); clear v i

% Target box width and height
w = coords(:,3) - coords(:,1);
h = coords(:,4) - coords(:,2);

% stats for outlier bbox rejection
[class_mu, class_sigma, class_cov] = per_class_stats(classes,w,h);
[~,~,~,n] = fcnunique(classes(:));
weights = 1./n(:)';  weights=weights/sum(weights);
vpa(n(:)')

% image weights (1395 does not exist, remove it)
image_weights = accumarray(chip_id,weights(xview_classes2indices(classes)+1),[847, 1]);
%i=uchips_numeric ~= 1395; 
image_weights = image_weights./sum(image_weights);
image_numbers = uchips_numeric;
%fig; bar(uchips_numeric(i), image_weights)

% K-means normalized with and height for 9 points
C = fcn_kmeans([w h], 30);
[~, i] = sort(C(:,1).*C(:,2));
C = C(i,:)';

% image mean and std
i = ~all(stats==0,2);
shapes=shapes(i,:);
stats=stats(i,:);  % rgb_mean, rgb_std
stat_means = zeros(1,12);
for i=1:12
    stat_means(i) = mean(fcnsigmarejection(stats(:,i),6,3));
end

% output RGB stats (comes in BGR from cv2.imread)
rgb_mu = stat_means([3 2 1])   % dataset RGB mean
rgb_std = stat_means([6 5 4])  % dataset RGB std mean
hsv_mu = stat_means(7:9)   % dataset RGB mean
hsv_std = stat_means(10:12)  % dataset RGB std mean
anchor_boxes = vpa(C(:)',4)  % anchor boxes

wh = single([image_w, image_h]);
classes = xview_classes2indices(classes);
targets = single([classes(:), coords]);
id = single(chip_number);  numel(id)
save('targets_c60NEW.mat','wh','targets','id','class_mu','class_sigma','class_cov','image_weights','image_numbers','-mat7-binary')










