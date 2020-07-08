input_square = linspace(0, 63, 64);
% Transpose required
input_square = reshape(input_square, [8, 8])';

input_small = linspace(0, 15, 16);
input_small = reshape(input_small, [4, 4])';

input_topleft = zeros(4, 4);
input_topleft(1, 1) = 100;
input_topleft(2, 1) = 10;
input_topleft(1, 2) = 1;
input_topleft(1, 4) = 100;

input_bottomright = zeros(4, 4);
input_bottomright(4, 4) = 100;
input_bottomright(3, 4) = 10;
input_bottomright(4, 3) = 1;
input_bottomright(4, 1) = 100;

fprintf('(8, 8) to (3, 4) without AA\n');
down_down_noaa = imresize( ...
    input_square, [3, 4], 'bicubic', 'antialiasing', false ...
);
save('down_down_noaa.mat', 'down_down_noaa');

fprintf('(8, 8) to (5, 7) without AA\n');
down_down_irregular_noaa = imresize( ...
    input_square, [5, 7], 'bicubic', 'antialiasing', false ...
);
save('down_down_irregular_noaa.mat', 'down_down_irregular_noaa');

fprintf('(4, 4) topleft to (5, 5) without AA\n');
up_up_topleft_noaa = imresize( ...
    input_topleft, [5, 5], 'bicubic', 'antialiasing', false ...
);
save('up_up_topleft_noaa.mat', 'up_up_topleft_noaa');
%disp(input_topleft);
%disp(up_up_topleft);

fprintf('(4, 4) bottomright to (5, 5) without AA\n');
up_up_bottomright_noaa = imresize( ...
    input_bottomright, [5, 5], 'bicubic', 'antialiasing', false ...
);
save('up_up_bottomright_noaa.mat', 'up_up_bottomright_noaa');
%disp(input_bottomright);
%disp(up_up_bottomright);

fprintf('(8, 8) to (11, 13) without AA\n');
up_up_irregular_noaa = imresize( ...
    input_square, [11, 13], 'bicubic', 'antialiasing', false ...
);
save('up_up_irregular_noaa.mat', 'up_up_irregular_noaa');

butterfly = imread(fullfile('..', 'example', 'butterfly.png'));
butterfly = im2double(butterfly);
fprintf('(256, 256) butterfly.png to (123, 234) without AA\n')
down_down_butterfly_irregular_noaa = imresize( ...
    butterfly, [123, 234], 'bicubic', 'antialiasing', false ...
);
save( ...
    'down_down_butterfly_irregular_noaa.mat', ...
    'down_down_butterfly_irregular_noaa' ...
);

fprintf('(256, 256) butterfly.png to (1234, 789) without AA\n')
up_up_butterfly_irregular_noaa = imresize( ...
    butterfly, [1234, 789], 'bicubic', 'antialiasing', false ...
);
save( ...
    'up_up_butterfly_irregular_noaa.mat', ...
    'up_up_butterfly_irregular_noaa' ...
);

