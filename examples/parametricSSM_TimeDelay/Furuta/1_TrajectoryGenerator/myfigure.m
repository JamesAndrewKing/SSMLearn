function fig = myfigure()
    set(groot, 'defaultTextInterpreter', 'latex');
    set(groot, 'defaultLegendInterpreter', 'latex');
    set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
    
    fig = figure('Name', '', ...
                 'Color', 'w', 'Position', [100, 100, 3600, 3600]);
             
    ax = axes('Parent', fig);
    ax.FontSize = 20;            
    ax.TickLabelInterpreter = 'latex';
    
    xlabel(ax, '', 'Interpreter', 'latex', 'FontSize', 20);
    ylabel(ax, '', 'Interpreter', 'latex', 'FontSize', 20);
    title(ax, '', 'Interpreter', 'latex', 'FontSize', 25);
end 