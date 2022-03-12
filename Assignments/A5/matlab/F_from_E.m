function F = F_from_E(E, K)
    K_inv = inv(K);
    F = K_inv'*E*K_inv;
end
