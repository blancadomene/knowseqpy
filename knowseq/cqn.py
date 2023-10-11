import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

from sklearn.preprocessing import quantile_transform


def cqn(counts, x, lengths, sizeFactors=None, subindex=None,
        tau=0.5, sqn=True, lengthMethod="smooth",
        verbose=False):
    lengthMethod = lengthMethod.lower()
    if len(lengths) == 1:
        lengths = [lengths[0] for _ in range(counts.shape[0])]
    if counts.ndim == 1:
        counts = counts.reshape((-1, 1))
    if counts.shape[0] != len(x) or counts.shape[0] != len(lengths):
        raise Exception(
            "arguments 'counts' need to have the same number of rows as the length of arguments 'x' and 'lengths'")
    if any(l <= 0 for l in lengths):
        raise Exception("argument 'lengths' need to be greater than zero")
    if sizeFactors is not None and len(sizeFactors) != counts.shape[1]:
        raise Exception(
            "argument 'sizeFactors' (when used) needs to have the same length as the number of columns of argument 'counts'")
    if subindex is not None and (min(subindex) <= 0 or max(subindex) > counts.shape[0]):
        raise Exception(
            "argument 'subindex' (when used) needs to be indices into the number of rows of argument 'counts'")

    if sizeFactors is None:
        sizeFactors = np.sum(counts, axis=0)

    if subindex is None:
        subindex = np.where(np.mean(counts, axis=1) > 50)[0]

    y = np.log2(counts + 1) - np.log2(sizeFactors / 10 ** 6).reshape((1, -1))
    if lengthMethod == "fixed":
        y -= np.log2(lengths / 1000).reshape((-1, 1))
    yfit = y[subindex, :]

    def fixPredictor(zz):
        zz_fit = zz[subindex]
        knots = np.quantile(zz_fit, [0.025, 0.25, 0.50, 0.75, 0.975]) + [0.01, 0, 0, 0, -0.01]
        grid = np.linspace(min(zz_fit), max(zz_fit), num=101)
        zz_out = zz.copy()
        zz2 = zz[np.logical_not(np.isin(np.arange(len(zz)), subindex))]
        zz2[zz2 < min(zz_fit)] = min(zz_fit)
        zz2[zz2 > max(zz_fit)] = max(zz_fit)
        zz_out[np.logical_not(np.isin(np.arange(len(zz)), subindex))] = zz2
        return zz_fit, zz_out, knots, grid

    x1_fit, x1_out, x1_knots, x1_grid = fixPredictor(x)

    if lengthMethod == "smooth":
        x2_fit, x2_out, x2_knots, x2_grid = fixPredictor(np.log2(lengths / 1000))
        df_fit = pd.DataFrame({'x1': x1_fit, 'x2': x2_fit})
        df_predict = pd.DataFrame({'x1': x1_out, 'x2': x2_out})
        df_func = pd.concat([
            pd.DataFrame({'x1': x1_grid, 'x2': [np.median(x2_grid)] * len(x1_grid)}),
            pd.DataFrame({'x1': [np.median(x1_grid)] * len(x2_grid), 'x2': x2_grid})
        ], axis=0)
    else:
        df_fit = pd.DataFrame({'x1': x1_fit})
        df_predict = pd.DataFrame({'x1': x1_out})
        df_func = pd.DataFrame({'x1': x1_grid})

    if verbose: print("RQ fit ", end="")
    regr = []
    for ii in range(yfit.shape[1]):
        if verbose: print(".", end="")
        df_fit['y'] = yfit[:, ii]
        if lengthMethod == "smooth":
            fit = smf.quantreg('y ~ x1 + x2', df_fit).fit(q=tau)
        else:
            fit = smf.quantreg('y ~ x1', df_fit).fit(q=tau)
        fitted = fit.predict(df_predict)
        func = fit.predict(df_func)
        regr.append({'fitted': fitted, 'func': func, 'coef': fit.params})
    if verbose: print()

    fitted = np.column_stack([r['fitted'] for r in regr])
    func = np.column_stack([r['func'] for r in regr])

    k = np.argsort(x[subindex])[len(subindex) // 2]
    offset0 = np.median(fitted[subindex[k], :])

    residuals = y - fitted
    if sqn:
        if verbose: print("SQN ", end="")
        residualsSQN = quantile_transform(residuals, n_quantiles=1000, output_distribution='normal', copy=True)
        if verbose: print(".")
        offset = residualsSQN + offset0 - y
    else:
        offset = residuals + offset0 - y

    glm_offset = offset + np.log2(sizeFactors / 10 ** 6).reshape((1, -1))
    glm_offset *= np.log(2)

    if lengthMethod == "smooth":
        func1 = func[:len(x1_grid), :]
        func2 = func[len(x1_grid):, :]
    else:
        func1 = func
        func2 = None

    return {
        'counts': counts, 'lengths': lengths, 'sizeFactors': sizeFactors,
        'subindex': subindex,
        'y': y, 'x': x, 'offset': offset, 'offset0': offset0,
        'glm.offset': glm_offset,
        'func1': func1, 'func2': func2,
        'grid1': x1_grid, 'grid2': x2_grid if lengthMethod == "smooth" else None,
        'knots1': x1_knots, 'knots2': x2_knots if lengthMethod == "smooth" else None
    }
