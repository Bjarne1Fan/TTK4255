function T_all = decompose_E(E)
    % Computes the four possible decompositions of E into a
    % relative pose, as described in Szeliski 7.2.
    %
    % The individual 4x4 matrices can be accessed as
    % T_i = T_all(:,:,i)

    [U,~,V] = svd(E);
    R90 = [0 -1 0 ; +1 0 0 ; 0 0 1];
    R1 = U*R90*V';
    R2 = U*R90'*V';
    if det(R1) < 0
        R1 = -R1;
    end
    if det(R2) < 0
        R2 = -R2;
    end
    t1 = U(:,3);
    t2 = -U(:,3);
    T_all = zeros(4,4,4);
    T_all(:,:,1) = [R1 t1 ; 0 0 0 1];
    T_all(:,:,2) = [R1 t2 ; 0 0 0 1];
    T_all(:,:,3) = [R2 t1 ; 0 0 0 1];
    T_all(:,:,4) = [R2 t2 ; 0 0 0 1];
end
