%% Machine Learning Online Class
%  Exercise 8 | Anomaly Detection and Collaborative Filtering
%
%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  exercise. You will need to complete the following functions:
%
%     estimateGaussian.m
%     selectThreshold.m
%     cofiCostFunc.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% ================== Part 1: Load Example Dataset  ===================
%  We start this exercise by using a small dataset that is easy to
%  visualize.
%
%  Our example case consists of 2 network server statistics across
%  several machines: the latency and throughput of each machine.
%  This exercise will help us find possibly faulty (or very fast) machines.
%

fprintf('Visualizing example dataset for outlier detection.\n\n');

%  The following command loads the dataset. You should now have the
%  variables X, Xval, yval in your environment
%load('ex8data1.mat');

dat = csvread('mydata.csv');

%  Visualize the example dataset
%plot(dat(:, 1), dat(:, 2), 'bx');
%axis([0 150 0 150]);
%pause
cols = dat(:, :);

%col2 = dat(3, :)
%axis([0 150 0 150]);
%xlabel('x');
%ylabel('y');
%plot(cols(2, :));
%fprintf('Program paused. Press enter to continue.\n');

%pause


%% ================== Part 2: Estimate the dataset statistics ===================
%  For this exercise, we assume a Gaussian distribution for the dataset.
%
%  We first estimate the parameters of our assumed Gaussian distribution,
%  then compute the probabilities for each of the points and then visualize
%  both the overall distribution and where each of the points falls in
%  terms of that distribution.
%
fprintf('Visualizing Gaussian fit.\n\n');

%  Estimate my and sigma2
%  R(p, mu, sigma2)
R = ones(3, size(cols));
%R
%c = 0;
for c = 1:size(cols)
    %[R(c, 2) R(c, 3)] = estimateGaussian(dat(c, :));
    [mu sigma2] = estimateGaussian(cols(c, :));
    p = multivariateGaussian(cols(c, :), mu, sigma2);
    R(2, c) = mu;
    R(3, c) = sigma2;
    R(1, c) = p;
    %fprintf('mu(%d)=>%f\n', c, mu);
    %fprintf('sigma2(%d) =>%f\n', c, sigma2);
end
%  Returns the density of the multivariate normal at each data point (row)
%  of X

fprintf('-----> Display R(p, mu, sigma2) :\n\n');
%R([1,:],:) = []
R(:, [1,:]) = [];
disp(R);
xlabel('x');
ylabel('y');
axis([0 1 0 1]);
plot(R(1, :));
pause
