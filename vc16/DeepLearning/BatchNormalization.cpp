#include "stdafx.h"
#include "BatchNormalization.h"

namespace mll
{
	batchnorm::batchnorm() : act(_act), momentum(_momentum)
	{
		// Set an object
		setObject();
	}

	batchnorm::batchnorm(const bool act, const double momentum) : act(_act), momentum(_momentum)
	{
		// Set an object
		setObject();

		// Set the parameters
		set(act, momentum);
	}

	batchnorm::batchnorm(const batchnorm& obj) : act(_act), momentum(_momentum)
	{
		// Set an object
		setObject();

		// Clone the object
		*this = obj;
	}

	batchnorm::~batchnorm()
	{
		// Clear the object
		clearObject();
	}

	batchnorm& batchnorm::operator=(const batchnorm& obj)
	{
		// Clear the object
		clearObject();

		// Set an object
		setObject();

		// Copy the object
		copyObject(obj);

		return *this;
	}

	void batchnorm::setObject()
	{
		// Set the parameters
		setType(*this);
		_act = false;
		_momentum = 0.9;
	}

	void batchnorm::copyObject(const object& obj)
	{
		// Do down casting
		const batchnorm* _obj = static_cast<const batchnorm*>(&obj);

		// Copy the parameters
		_act = _obj->_act;
		_momentum = _obj->_momentum;

		// Copy the memories
		mu_p = _obj->mu_p;
		var_p = _obj->var_p;
		gamma = _obj->gamma;
		beta = _obj->beta;
		mu = _obj->mu;
		var = _obj->var;
		std = _obj->std;
		X = _obj->X;
		X_norm = _obj->X_norm;
		X_mu = _obj->X_mu;
		dX_norm = _obj->dX_norm;
		dvar = _obj->dvar;
		stdi = _obj->stdi;
		dmu = _obj->dmu;
		dgamma = _obj->dgamma;
		dbeta = _obj->dbeta;
	}

	void batchnorm::clearObject()
	{
		// Clear the memories
		mu_p.release();
		var_p.release();
		gamma.release();
		beta.release();
		mu.release();
		var.release();
		std.release();
		X.release();
		X_norm.release();
		X_mu.release();
		dX_norm.release();
		dvar.release();
		stdi.release();
		dmu.release();
		dgamma.release();
		dbeta.release();
	}

	void batchnorm::set(const bool act, const double momentum)
	{
		// Set the parameters
		_act = act;
		_momentum = momentum;
	}

	const numat batchnorm::forward(const numat& net)
	{
		// Check the activation flag
		if (_act == false)
		{
			return net;
		}

		// Check the memory size
		if (X.rows != net.rows)
		{
			// Create the memories
			X = numat::zeros(msize(net.rows, net.cols));
			X_norm = numat::zeros(msize(net.rows, net.cols));
		}
		else
		{
			// Copy the memory
			for (int i = 0; i < net.length(); i++)
			{
				X(i) = net(i);
			}
		}

		// Check the memory status
		if (mu.empty() == true || var.empty() == true || std.empty() == true)
		{
			mu = numat::zeros(msize(1, net.cols));
			var = numat::zeros(msize(1, net.cols));
			std = numat::zeros(msize(1, net.cols));
			mu_p = numat::zeros(msize(1, net.cols));
			var_p = numat::ones(msize(1, net.cols));
			gamma = numat::ones(msize(1, net.cols));
			beta = numat::zeros(msize(1, net.cols));
		}
		else
		{
			// Initialize the memories
			for (int i = 0; i < net.cols; i++)
			{
				mu(i) = 0.0;
				var(i) = 0.0;
			}
		}

		// Activative the data
		numat out(msize(net.rows, net.cols));
		for (int i = 0; i < net.cols; i++)
		{
			// Calculate a mean value
			for (int j = 0; j < net.rows; j++)
			{
				mu(i) += net(j, i);
			}
			mu(i) /= net.rows;

			// Calculate a variance value
			for (int j = 0; j < net.rows; j++)
			{
				var(i) += (net(j, i) - mu(i)) * (net(j, i) - mu(i));
			}
			var(i) /= net.rows;

			// Normalize the feature vector
			std(i) = sqrt(var(i) + 1e-8);
			for (int j = 0; j < net.rows; j++)
			{
				X_norm(j, i) = (net(j, i) - mu(i)) / std(i);
				out(j, i) = gamma(i) * X_norm(j, i) + beta(i);
			}

			// Calculate moving average values for the inference
			mu_p(i) = _momentum * mu_p(i) + (1.0 - _momentum) * mu(i);
			var_p(i) = _momentum * var_p(i) + (1.0 - _momentum) * var(i);
		}

		return out;
	}

	const numat batchnorm::backward(const numat& dout)
	{
		// Check the activation flag
		if (_act == false)
		{
			return dout;
		}

		// Check the memory size
		if (X_mu.rows != dout.rows)
		{
			// Create the memories
			X_mu = numat::zeros(msize(dout.rows, dout.cols));
			dX_norm = numat::zeros(msize(dout.rows, dout.cols));
		}
		else
		{
			// Initialize the memories
			for (int i = 0; i < dout.length(); i++)
			{
				X_mu(i) = 0.0;
				dX_norm(i) = 0.0;
			}
		}

		// Check the memory status
		if (dmu.empty() == true || dvar.empty() == true || stdi.empty() == true)
		{
			// Create the memories
			dmu = numat::zeros(msize(mu.rows, mu.cols));
			dvar = numat::zeros(msize(var.rows, var.cols));
			stdi = numat::zeros(msize(var.rows, var.cols));
			dgamma = numat::ones(msize(gamma.rows, gamma.cols));
			dbeta = numat::ones(msize(beta.rows, beta.cols));
		}
		else
		{
			// Initialize the memories
			for (int i = 0; i < gamma.cols; i++)
			{
				dmu(i) = 0.0;
				dvar(i) = 0.0;
				stdi(i) = 0.0;
				dgamma(i) = 0.0;
				dbeta(i) = 0.0;
			}
		}

		// Derivative the data
		numat deriv(msize(dout.rows, dout.cols));
		for (int i = 0; i < dout.cols; i++)
		{
			// Calculate the derivatives on the normalized X
			for (int j = 0; j < dout.rows; j++)
			{
				dX_norm(j, i) = dout(j, i) * gamma(i);
			}

			// Calculate the derivatives on the variance
			stdi(i) = 1.0 / sqrt(var(i) + 1e-8);
			for (int j = 0; j < dout.rows; j++)
			{
				X_mu(j, i) = X(j, i) - mu(i);
				dvar(i) += dX_norm(j, i) * X_mu(j, i) * (-0.5) * pow(stdi(i), 3.0);
			}

			// Calculate the derivatives on the mean
			double temp = 0.0;
			for (int j = 0; j < dout.rows; j++)
			{
				dmu(i) += dX_norm(j, i) * (-stdi(i));
				temp += -2.0 * X_mu(j, i);
			}
			dmu(i) += dvar(i) * temp / dout.rows;

			// Calculate the derivatives on the X
			for (int j = 0; j < dout.rows; j++)
			{
				deriv(j, i) = (dX_norm(j, i) * stdi(i)) + (dvar(i) * 2.0 * X_mu(j, i) / dout.rows) + (dmu(i) / dout.rows);
			}

			// Calculate the derivatives on the gamma
			for (int j = 0; j < dout.rows; j++)
			{
				dgamma(i) += dout(j, i) * X_norm(j, i);
			}

			// Calculate the derivatives on the beta
			for (int j = 0; j < dout.rows; j++)
			{
				dbeta(i) += dout(j, i);
			}
		}

		return deriv;
	}

	void batchnorm::update(const double epsilon)
	{
		// Check the activation flag
		if (_act == true)
		{
			// Update the scale parameters
			for (int i = 0; i < gamma.length(); i++)
			{
				gamma(i) -= epsilon * dgamma(i);
				beta(i) -= epsilon * dbeta(i);
			}
		}
	}

	const numat batchnorm::inference(const numat& net)
	{
		// Check the activation flag
		if (_act == false)
		{
			return net;
		}

		// Check the memory status
		if (mu_p.empty() == true || var_p.empty() == true || gamma.empty() == true || beta.empty() == true)
		{
			// Create Memories
			mu_p = numat::zeros(msize(1, net.cols));
			var_p = numat::ones(msize(1, net.cols));
			gamma = numat::ones(msize(1, net.cols));
			beta = numat::zeros(msize(1, net.cols));
		}

		// Normalize the feature vector
		numat out(msize(net.rows, net.cols));
		for (int i = 0; i < net.cols; i++)
		{
			const double std_p = sqrt(var_p(i) + 1e-8);
			for (int j = 0; j < net.rows; j++)
			{
				out(j, i) = gamma(i) * ((net(j, i) - mu_p(i)) / std_p) + beta(i);
			}
		}

		return out;
	}
}