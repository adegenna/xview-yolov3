function [] = make_small_chips()
clc; close; clear
load('targets_c60NEW.mat')
path_a = '/home/adegennaro/Projects/OGA/cat/data/xview/train/images/';

% rmdir([path_a 'classes'],'s')
% for i=0:59
%     mkdir(sprintf([path_a 'classes/%g'],i))
% end

uid = unique(id)';
class_count = zeros(1,60);
f_count = 0;  % file count
c_count = 0;  % chip count
length = 128;  % combined inner + padding
lengh_inner = 64;  % core size
X = zeros(150000,length,length,3, 'uint8');
Y = zeros(1,150000,'uint8');
for i = uid
    f_count = f_count+1;
    fprintf('%g/847\n',f_count)
    target_idx = find(id==i)';
    img = imread(sprintf([path_a 'train_images_%g.tif'],i));

    %fig(4,4)
    for j = target_idx
        t = targets(j,:); %#ok<*NODEF>
        class = t(1);
        
        if any(class == [5, 48]) && rand>0.1  % skip 90% of buildings and cars
            continue
        end
        
        x1=t(2)+1;  y1=t(3)+1;  x2=t(4)+1;  y2=t(5)+1;
        class_count(class+1) = class_count(class+1) + 1;
        w = x2-x1;  h = y2-y1;
        xc = (x2 + x1)/2;  yc = (y2 + y1)/2;
        image_wh = wh(j,:);
        
        % make chip a square
        l = round((max(w,h)*1.0 + 2) * length/lengh_inner) / 2;  % normal
        
        lx = floor(min(min(xc-1, l), image_wh(1)-xc));
        ly = floor(min(min(yc-1, l), image_wh(2)-yc));
        
        img1 = img(round(yc-ly):round(yc+ly), round(xc-lx):round(xc+lx), :);
        img2 = imresize(img1,[length length], 'bilinear');
        
        c_count = c_count + 1;
        Y(c_count) = class;
        X(c_count,:,:,:) = img2;

        % imwrite(img2,sprintf([path_a 'classes/%g/%g.bmp'],class,class_count(class+1)));
        % sca; imshow(img2); axis equal ij; title([num2str(class) ' - ' xview_names(class)])
        
        %if mod(c_count,16)==0
        %    ''
        %end
    end
end

% X = permute(X(1:c_count,:,:,:),[1 4 2 3]); %#ok<*NASGU> permute to pytorch standards
X = X(1:c_count,:,:,:);
Y = Y(1:c_count);

X = permute(X,[4,3,2,1]);  % for hd5y only (reads in backwards permuted)
save('-v7.3','class_chips64+64_tight','X','Y')

% 32 + 14 = 46
% 40 + 16 = 56
% 48 + 16 = 64
% 64 + 26 = 90
end
