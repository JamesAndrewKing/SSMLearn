function fig = MyCustomFigure(varargin)
% fig = MyCustomFigure(...)
% Open a figure with customizable global settings for SSMLearn.
% Allows specification of figure number, subplot layout, interpreter,
% font sizes (axes, labels, title), font name, background color, grid,
% box, axes line width, plot line width, and screen coverage.
%
% Name-Value Inputs:
%   'FigNumber'        : integer >=1 (default: new figure)
%   'SubPlot'          : [rows,cols] vector for subplot grid (default: none)
%   'Interpreter'      : 'latex' or 'none' (default: 'none')
%   'FontName'         : e.g. 'Helvetica' (default: 'Helvetica')
%   'FontSizeAxes'     : font size for tick labels (default: 16)
%   'FontSizeLabels'   : font size for axis labels (default: 16)
%   'FontSizeTitle'    : font size for titles (default: 18)
%   'AxesLineWidth'    : numeric line width for axes box (default: 0.5)
%   'PlotLineWidth'    : numeric default line width for plotted data (default: 0.5)
%   'BackgroundColor'  : e.g. 'white' (default: 'white')
%   'Grid'             : 'on' or 'off' (default: 'on')
%   'Box'              : 'on' or 'off' (default: 'on')
%   'ScreenCoverage'   : fraction of screen to cover [0,1] (default: 0.8)
%
% OUTPUT:
%   fig : figure handle

% Default settings
defaults = struct( ...
    'FigNumber',        [], ...
    'SubPlot',          [], ...
    'Interpreter',      'none', ...
    'FontName',         'Helvetica', ...
    'FontSizeAxes',     16, ...
    'FontSizeLabels',   16, ...
    'FontSizeTitle',    18, ...
    'AxesLineWidth',    0.5, ...
    'PlotLineWidth',    0.5, ...
    'BackgroundColor',  'white', ...
    'Grid',             'on', ...
    'Box',              'on', ...
    'ScreenCoverage',   0.8  ...
);

% Parse inputs
p = inputParser;
addParameter(p, 'FigNumber',       defaults.FigNumber);
addParameter(p, 'SubPlot',         defaults.SubPlot);
addParameter(p, 'Interpreter',     defaults.Interpreter);
addParameter(p, 'FontName',        defaults.FontName);
addParameter(p, 'FontSizeAxes',    defaults.FontSizeAxes);
addParameter(p, 'FontSizeLabels',  defaults.FontSizeLabels);
addParameter(p, 'FontSizeTitle',   defaults.FontSizeTitle);
addParameter(p, 'AxesLineWidth',   defaults.AxesLineWidth);
addParameter(p, 'PlotLineWidth',   defaults.PlotLineWidth);
addParameter(p, 'BackgroundColor', defaults.BackgroundColor);
addParameter(p, 'Grid',            defaults.Grid);
addParameter(p, 'Box',             defaults.Box);
addParameter(p, 'ScreenCoverage',  defaults.ScreenCoverage);
parse(p, varargin{:});
opts = p.Results;

% Create figure
if isempty(opts.FigNumber)
    fig = figure('Color', opts.BackgroundColor);
else
    fig = figure(opts.FigNumber);
    set(fig, 'Color', opts.BackgroundColor);
    clf(fig);
end

% Set default line width for plotted data
set(fig, 'defaultLineLineWidth', opts.PlotLineWidth);

% Adjust figure size to cover specified fraction of the screen
scr = get(0, 'ScreenSize');            % [left bottom width height]
cov = max(0, min(1, opts.ScreenCoverage));  % clamp between 0 and 1
w = scr(3) * cov;
h = scr(4) * cov;
x0 = scr(1) + (scr(3) - w) / 2;
y0 = scr(2) + (scr(4) - h) / 2;
set(fig, 'Units', 'pixels', 'Position', [x0, y0, w, h]);

% Apply subplots or single axes
if isempty(opts.SubPlot)
    ax = gca;
    hold(ax, 'on');
    grid(ax, opts.Grid);
    box(ax, opts.Box);
    applyStyling(ax, opts);
else
    rows = opts.SubPlot(1);
    cols = opts.SubPlot(2);
    for idx = 1:rows*cols
        ax = subplot(rows, cols, idx);
        hold(ax, 'on');
        grid(ax, opts.Grid);
        box(ax, opts.Box);
        applyStyling(ax, opts);
    end
end

end

function applyStyling(ax, opts)
% Apply font, interpreter, and axes line width settings to an axes
set(ax, 'FontName', opts.FontName);
set(ax, 'FontSize', opts.FontSizeAxes);
set(ax, 'LineWidth', opts.AxesLineWidth);
if strcmpi(opts.Interpreter, 'latex')
    set(ax, 'DefaultTextInterpreter', 'latex');
    set(ax, 'TickLabelInterpreter', 'latex');
else
    set(ax, 'DefaultTextInterpreter', 'none');
    set(ax, 'TickLabelInterpreter', 'none');
end
% Set default label/title font sizes
set(ax.Title,  'FontSize', opts.FontSizeTitle);
set(ax.XLabel, 'FontSize', opts.FontSizeLabels);
set(ax.YLabel, 'FontSize', opts.FontSizeLabels);
end
