function [coords, valid] = clean_coords(chip_number, coords, classes, image_h, image_w)
% j = chip_number == 2294;
% coords = coords(j,:);
% classes = classes(j);
% image_h = image_h(j,:);
% image_w = image_w(j,:);

x1 = coords(:,1);
y1 = coords(:,2);
x2 = coords(:,3);
y2 = coords(:,4);
w = x2-x1;
h = y2-y1;
area = w.*h;

% crop
x1 = min( max(x1,0), image_w);
y1 = min( max(y1,0), image_h);
x2 = min( max(x2,0), image_w);
y2 = min( max(y2,0), image_h);
w = x2-x1;
h = y2-y1;
new_area = w.*h;
new_ar = max(w./h, h./w);
coords = [x1 y1 x2 y2];

% no nans or infs in bounding boxes
i0 = ~any(isnan(coords) | isinf(coords), 2);

% % sigma rejections on dimensions (entire dataset)
% [~, i1] = fcnsigmarejection(new_area,21, 3);
% [~, i2] = fcnsigmarejection(w,21, 3);
% [~, i3] = fcnsigmarejection(h,21, 3);
i1 = true(size(w));
i2 = true(size(w));
i3 = true(size(w));

% sigma rejections on dimensions (per class)
uc=unique(classes(:));
for i = 1:numel(uc)
    j = find(classes==uc(i));
    [~,v] = fcnsigmarejection(new_area(j),12,3);    i1(j) = i1(j) & v;
    [~,v] = fcnsigmarejection(w(j),12,3);           i2(j) = i2(j) & v;
    [~,v] = fcnsigmarejection(h(j),12,3);           i3(j) = i3(j) & v;
end

% manual dimension requirements
i4 = new_area >= 20 & w > 4 & h > 4 & new_ar<15;  

% extreme edges (i.e. don't start an x1 10 pixels from the right side)
i5 = x1 < (image_w-10) & y1 < (image_h-10) & x2 > 10 & y2 > 10;  % border = 5

% cut objects that lost >90% of their area during crop
new_area_ratio = new_area./ area;
i6 = new_area_ratio > 0.25;

% no image dimension nans or infs, or smaller than 32 pix
hw = [image_h image_w];
i7 = ~any(isnan(hw) | isinf(hw) | hw < 32, 2);

% remove invalid classes 75 and 82 ('None' class, wtf?)
i8 = ~any(classes(:) == [75, 82],2);

% remove 18 and 73 (small cars and buildings) as an experiment
%i9 = any(classes(:) == [11, 12, 13, 15, 74, 84],2);  % group 0 aircraft
%i9 = any(classes(:) == [17, 18, 19],2);  % group 1 cars
%i9 = any(classes(:) == [71, 72, 76, 77, 79, 83, 86, 89, 93, 94],2);  % group 2 buildings
%i9 = any(classes(:) == [20, 21, 23, 24, 25, 26, 27, 28, 29, 32, 60, 91],2);  % group 3 trucks
%i9 = any(classes(:) == [33, 34, 35, 36, 37, 38],2);  % group 4 trains
%i9 = any(classes(:) == [40, 41, 42, 44, 45, 47, 49, 50, 51, 52],2);  % group 5 boats
%i9 = any(classes(:) == [53, 54, 55, 56, 57, 59, 61, 62, 63, 64, 65, 66],2);  % group 6 docks

valid = i0 & i1 & i2 & i3 & i4 & i5 & i6 & i7 & i8;
coords = coords(valid,:);
% fig; imshow(x2294); plot(x1,y1,'.'); plot(x2,y2,'.')
end      
