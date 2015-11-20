#include <Eigen/Core>
#include <Eigen/Cholesky>
#include "logistic_ridge.h"
#include "logistic_ridge_newton.h"

typedef Eigen::MatrixXd Matrix;
typedef Eigen::VectorXd Vector;
typedef Eigen::Map<Matrix> MapMat;
typedef Eigen::Map<Vector> MapVec;

void logistic_ridge(double *x_, int n, int p, double *y_,
                    double lambda, double *v_,
                    int max_iter, double eps_abs, double eps_rel,
                    int *niter, double *coef_)
{
    MapMat x(x_, n, p);
    MapVec y(y_, n);
    Matrix H0(p, p);
    MapVec v(v_, p);
    MapVec coef(coef_, p);

    H0.triangularView<Eigen::Lower>() = x.transpose() * x * 0.25;
    Eigen::LLT<Matrix> solver(H0 + lambda * Matrix::Identity(p, p));

    int i;
    coef.setZero();
    Vector mu(n);
    Vector grad(p);
    Vector delta(p);
    for(i = 0; i < max_iter; i++)
    {
        mu.noalias() = x * coef;
        for(int j = 0; j < n; j++)
            mu[j] = 1.0 / (1.0 + std::exp(-mu[j]));
        grad.noalias() = x.transpose() * (mu - y) + lambda * (coef - v);
        delta.noalias() = solver.solve(grad);
        coef.noalias() -= delta;
        double r = delta.norm();
        if(r < eps_abs * std::sqrt(double(p)) || r < eps_rel * coef.norm())
            break;
    }

    *niter = i;
}

void logistic_ridge_newton(double *x_, int n, int p, double *y_,
                           double lambda, double *v_,
                           int max_iter, double eps_abs, double eps_rel,
                           int *niter, double *coef_)
{
    MapMat x(x_, n, p);
    MapVec y(y_, n);
    MapVec v(v_, p);
    MapVec coef(coef_, p);

    int i;
    coef.setZero();
    Matrix wx(n, p);
    Matrix H(p, p);
    Vector mu(n);
    Vector grad(p);
    Vector delta(p);
    for(i = 0; i < max_iter; i++)
    {
        // mu = 1 / (1 + exp(-x * b))
        mu.noalias() = x * coef;
        for(int j = 0; j < n; j++)
            mu[j] = 1.0 / (1.0 + std::exp(-mu[j]));

        grad.noalias() = x.transpose() * (mu - y) + lambda * (coef - v);
        // Calculate wsqrt = sqrt(mu * (1 - mu))
        for(int j = 0; j < n; j++)
            mu[j] = std::sqrt(mu[j] * (1 - mu[j]));
        // Multiply each column of x by wsqrt
        for(int j = 0; j < p; j++)
            wx.col(j).noalias() = x.col(j).cwiseProduct(mu);
        // Calculate X'WX
        H.triangularView<Eigen::Lower>() = wx.transpose() * wx;
        H.diagonal().array() += lambda;
        Eigen::LLT<Matrix, Eigen::Lower> solver(H);

        delta.noalias() = solver.solve(grad);
        coef.noalias() -= delta;
        double r = delta.norm();
        if(r < eps_abs * std::sqrt(double(p)) || r < eps_rel * coef.norm())
            break;
    }

    *niter = i;
}



JNIEXPORT jdoubleArray JNICALL Java_statr_stat598bd_LogisticRidgeNative_logistic_1ridge
  (JNIEnv *env, jobject obj, jdoubleArray jx, jint n, jint p,
   jdoubleArray jy, jdouble lambda, jdoubleArray jv,
   jint max_iter, jdouble eps_abs, jdouble eps_rel, jintArray niter)
{
    jdouble *x = env->GetDoubleArrayElements(jx, NULL);
    if(x == NULL)  return NULL;

    jdouble *y = env->GetDoubleArrayElements(jy, NULL);
    if(y == NULL)  return NULL;

    jdouble *v = env->GetDoubleArrayElements(jv, NULL);
    if(v == NULL)  return NULL;

    jdouble *coef = new jdouble[p];

    int iter;
    logistic_ridge(x, n, p, y, lambda, v, max_iter, eps_abs, eps_rel, &iter, coef);

    env->ReleaseDoubleArrayElements(jx, x, 0);
    env->ReleaseDoubleArrayElements(jy, y, 0);
    env->ReleaseDoubleArrayElements(jv, v, 0);

    env->SetIntArrayRegion(niter, 0, 1, &iter);

    jdoubleArray res = env->NewDoubleArray(p);
    if(res == NULL)  return NULL;
    env->SetDoubleArrayRegion(res, 0, p, coef);
    delete [] coef;

    return res;
}

JNIEXPORT jdoubleArray JNICALL Java_statr_stat598bd_LogisticRidgeNewtonNative_logistic_1ridge_1newton
  (JNIEnv *env, jobject obj, jdoubleArray jx, jint n, jint p,
   jdoubleArray jy, jdouble lambda, jdoubleArray jv,
   jint max_iter, jdouble eps_abs, jdouble eps_rel, jintArray niter)
{
    jdouble *x = env->GetDoubleArrayElements(jx, NULL);
    if(x == NULL)  return NULL;

    jdouble *y = env->GetDoubleArrayElements(jy, NULL);
    if(y == NULL)  return NULL;

    jdouble *v = env->GetDoubleArrayElements(jv, NULL);
    if(v == NULL)  return NULL;

    jdouble *coef = new jdouble[p];

    int iter;
    logistic_ridge_newton(x, n, p, y, lambda, v, max_iter, eps_abs, eps_rel, &iter, coef);

    env->ReleaseDoubleArrayElements(jx, x, 0);
    env->ReleaseDoubleArrayElements(jy, y, 0);
    env->ReleaseDoubleArrayElements(jv, v, 0);

    env->SetIntArrayRegion(niter, 0, 1, &iter);

    jdoubleArray res = env->NewDoubleArray(p);
    if(res == NULL)  return NULL;
    env->SetDoubleArrayRegion(res, 0, p, coef);
    delete [] coef;

    return res;
}
