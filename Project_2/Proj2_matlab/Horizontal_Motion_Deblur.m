clear all
close all
clc

% number of rows of your reference image. Use a small number if your computer is slow
row = 100;
ori_img = im2double(imresize(rgb2gray(imread('C:\Users\BRUCE WANG\Desktop\origin.jpg')), [row NaN]));
figure
imshow(ori_img)
title('Reference Image')

x = reshape(ori_img,[],1);
n = size(x,1);
A = eye(n,n);

% out of focus motion matrix
fprintf('Sloving A matrix......\n');
for i = 1:n
    if i+1 < n
        A(i,i+1) = 1/3;
    end
    
    if i-1 > 0
        A(i,i-1) = 1/3;
    end
    A(i,i) = 1/3;
end
fprintf('sucess!!!\n');

b = A*x;
blur_img = reshape(b,size(ori_img));
figure
imshow(blur_img);
title('Horizontal Image');

deblur_img = reshape(Solving_Linear_Equations_with_LU_decomposition(A,b),size(ori_img));

if max(max(abs(deblur_img - ori_img))) > 1e-10
    error('wrong motion matrix');
end

figure
imshow(deblur_img);
title('Deblur Image');
fprintf('done!!!\n');



