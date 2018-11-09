function perf(T,logplot)
%PERF    Performace profiles
%
% PERF(T,logplot)-- produces a performace profile as described in
%   Benchmarking optimization software with performance profiles,
%   E.D. Dolan and J.J. More', 
%   Mathematical Programming, 91 (2002), 201--213.
% Each column of the matrix T defines the performance data for a solver.
% Failures on a given problem are represented by a NaN.
% The optional argument logplot is used to produce a 
% log (base 2) performance plot.
%
% This function is based on the perl script of Liz Dolan.
%
% Jorge J. More', June 2004

if (nargin < 2)
    logplot = 0; 
end

color  = ['m' 'b' 'r' 'g' 'c' 'k' 'y' [0 0.5 1]];
line   = [':' '-' '-.' '--' '-,' ':-' '.:' ';'];
marker = ['o' 's' '*' 'd' 'x' 'v' '^' 'o' '.' 'h'];

[np,ns] = size(T);

% Minimal performance per solver

minperf = min(T,[],2);

% Compute ratios and divide by smallest element in each row.

r = zeros(np,ns);
for p = 1: np
  r(p,:) = T(p,:)/minperf(p);
end

if (logplot)
    r = log2(r);
end

max_ratio = max(max(r));

% Replace all NaN's with twice the max_ratio and sort.

r(isnan(r)) = 2*max_ratio;
r = sort(r);

% Plot stair graphs with markers.

clf;
for s = 1: ns
 [xs,ys] = stairs(r(:,s),(1:np)/np);
  option = ['-' color(s) marker(s)];
  plot(xs,ys,option,'MarkerSize',4, 'LineWidth', 1);
%  option = [line(s)];
%  plot(xs,ys,option);

hold on;
 %__________my tweaks
  xlabel('{\tau}');
  ylabel('P({log_{2}(r_{p,s})} \leq \tau : 1\leq s\leq n_{s})')

end

% Axis properties are set so that failures are not shown,
% but with the max_ratio data points shown. This highlights
% the "flatline" effect.

%axis([ 0 1.1*max_ratio 0 1 ]);
legend('Threshold (n = 5)','Threshold (n = 10)','Threshold (n = 15)','Threshold (n = 20)')
title('Log_{2} Performance Profile plot for recognition algorithm at different threshold computation')
% Legends and title should be added.



