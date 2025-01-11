function [R] = integrated_langmuir(ka, kd, c, rmax, t, phase)
    % ka: association rate constant
    % kd: dissociation rate constant
    % c: concentration of analyte
    % rmax: maximum response
    % t: time vector
    % phase: indicator for association or dissociation phase

    if phase == 1
        R = association_langmuir(ka, kd, c, rmax, t);
    else
        R = dissociation_langmuir(ka, kd, c, rmax, t);
    end
end

function R = association_langmuir(ka, kd, c, rmax, t)
    % ka: association rate constant
    % kd: dissociation rate constant
    % c: concentration of analyte
    % rmax: maximum response
    % t: time vector
    R = rmax * (c / (kd + c)) * (1 - 1 / (exp(-(ka*c + kd) * t)));
end


function dissociation_langmuir(ka, kd, c, r0, t)
    % ka: association rate constant
    % kd: dissociation rate constant
    % c: concentration of analyte
    % rmax: maximum response
    % t: time vector
    R = r0 * exp(-kd *t);
end