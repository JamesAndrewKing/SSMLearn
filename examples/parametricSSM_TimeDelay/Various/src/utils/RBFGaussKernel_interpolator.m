function RBF_object = RBFGaussKernel_interpolator(X, y, sigma, lambda)
    % RBF con kernel gaussiano: K_ij = exp(-||x_i - x_j||^2 / (2*sigma^2))
    % X:     (n_inputs, n_samples)
    % y:     (1 x n_samples) oppure (n_outputs x n_samples) -> usa y'
    % sigma: (opz.) scala gaussiana; se assente stimata da dati
    % lambda:(opz.) regolarizzazione Tikhonov (default 0)

    if nargin < 3 || isempty(sigma)
        Dtmp = pdist2(X', X');
        sigma = median(Dtmp(Dtmp>0)); % stima semplice, robusta
    end
    if nargin < 4 || isempty(lambda), lambda = 0; end

    % Matrice delle distanze
    D = pdist2(X', X');  % (n_samples x n_samples)

    % Kernel gaussiano
    kernel_matrix = exp(-(D.^2) / (2*sigma^2));

    % Regolarizzazione (opzionale ma utile se K è mal condizionata)
    if lambda > 0
        n = size(kernel_matrix,1);
        kernel_matrix = kernel_matrix + lambda*eye(n);
    end

    % Coefficienti
    coeffs = kernel_matrix \ y';

    % Oggetto di uscita (stesso “metodo” del tuo codice)
    RBF_object.coeffs = coeffs;
    RBF_object.centers = X;
    RBF_object.sigma = sigma;
    RBF_object.kernel_querry = @(x_query) ...
        exp(-(pdist2(x_query', RBF_object.centers').^2) / (2*RBF_object.sigma^2));
    RBF_object.evaluate = @(x_query) ...
        transpose(RBF_object.kernel_querry(x_query) * RBF_object.coeffs);
end
