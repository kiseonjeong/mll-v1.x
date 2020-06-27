#include "stdafx.h"
#include "ActivationFunction.h"

namespace mll
{
	namespace nn
	{
		actfunc::actfunc() : type(_type), alpha(_alpha)
		{
			// Set an object
			setObject();
		}

		actfunc::~actfunc()
		{
			// Clear the object
			clearObject();
		}

		void actfunc::setObject()
		{
			// Set the parameters
			setType(*this);
			_type = ACT_FUNC_UNKNOWN;
		}

		void actfunc::copyObject(const object& obj)
		{
			// Do down casting
			actfunc* _obj = (actfunc*)&obj;

			// Copy the parameters
			_type = _obj->_type;
		}

		void actfunc::clearObject()
		{
			// Do nothing
		}

		identity::identity()
		{
			// Set an object
			setObject();
		}

		identity::identity(const identity& obj)
		{
			// Set an object
			setObject();

			// Clone the object
			*this = obj;
		}

		identity::~identity()
		{
			// Clear the object
			clearObject();
		}

		identity& identity::operator=(const identity& obj)
		{
			// Clear the object
			clearObject();

			// Set an object
			setObject();

			// Copy the object
			copyObject(obj);

			return *this;
		}

		void identity::setObject()
		{
			// Set the parameters
			setType(*this);
			_type = ACT_FUNC_IDENTITY;
			_alpha = 1.0;
		}

		void identity::copyObject(const object& obj)
		{
			// Do down casting
			identity* _obj = (identity*)&obj;

			// Copy the parameters
			_type = _obj->_type;
			_alpha = _obj->_alpha;

			// Copy the memories
			fx = _obj->fx;
			df = _obj->df;
			t = _obj->t;
		}

		void identity::clearObject()
		{
			// Release the memories
			fx.release();
			df.release();
			t.release();
		}

		void identity::set(const double alpha)
		{
			// Set the hyperparameter
			_alpha = alpha;
		}

		const numat identity::activative(const numat& x)
		{
			// Activative the data
			fx = x;

			return fx;
		}

		const numat identity::derivative()
		{
			// Derivative the data
			df = numat::ones(msize(fx.rows, fx.cols));

			return df;
		}

		linear::linear()
		{
			// Set an object
			setObject();
		}

		linear::linear(const double alpha)
		{
			// Set an object
			setObject();

			// Set the parameters
			set(alpha);
		}

		linear::linear(const linear& obj)
		{
			// Set an object
			setObject();

			// Clone the object
			*this = obj;
		}

		linear::~linear()
		{
			// Clear the object
			clearObject();
		}

		linear& linear::operator=(const linear& obj)
		{
			// Clear the object
			clearObject();

			// Set an object
			setObject();

			// Copy the object
			copyObject(obj);

			return *this;
		}

		void linear::setObject()
		{
			// Set the parameters
			setType(*this);
			_type = ACT_FUNC_LINEAR;
			_alpha = 1.0;
		}

		const numat linear::activative(const numat& x)
		{
			// Activative the data
			fx = _alpha * x;

			return fx;
		}

		const numat linear::derivative()
		{
			// Derivative the data
			df = numat::values(msize(fx.rows, fx.cols), _alpha);

			return df;
		}

		sigmoid::sigmoid()
		{
			// Set an object
			setObject();
		}

		sigmoid::sigmoid(const double alpha)
		{
			// Set an object
			setObject();

			// Set the parameters
			set(alpha);
		}

		sigmoid::sigmoid(const sigmoid& obj)
		{
			// Set an object
			setObject();

			// Clone the object
			*this = obj;
		}

		sigmoid::~sigmoid()
		{
			// Clear the object
			clearObject();
		}

		sigmoid& sigmoid::operator=(const sigmoid& obj)
		{
			// Clear the object
			clearObject();

			// Set an object
			setObject();

			// Copy the object
			copyObject(obj);

			return *this;
		}

		void sigmoid::setObject()
		{
			// Set Parameters
			setType(*this);
			_type = ACT_FUNC_SIGMOID;
			_alpha = 1.0;
		}

		const numat sigmoid::activative(const numat& x)
		{
			// Activative the data
			fx = 1.0 / (1.0 + numat::exp(-_alpha * x));

			return fx;
		}

		const numat sigmoid::derivative()
		{
			// Derivative the data
			df = (_alpha * fx).mul(1.0 - fx);

			return df;
		}

		tanh::tanh()
		{
			// Set an object
			setObject();
		}

		tanh::tanh(const double alpha)
		{
			// Set an object
			setObject();

			// Set the parameters
			set(alpha);
		}

		tanh::tanh(const tanh& obj)
		{
			// Set an object
			setObject();

			// Clone the object
			*this = obj;
		}

		tanh::~tanh()
		{
			// Clear the object
			clearObject();
		}

		tanh& tanh::operator=(const tanh& obj)
		{
			// Clear the object
			clearObject();

			// Set an object
			setObject();

			// Copy the object
			copyObject(obj);

			return *this;
		}

		void tanh::setObject()
		{
			// Set the parameters
			setType(*this);
			_type = ACT_FUNC_HYPER_TANGENT;
			_alpha = 1.0;
		}

		const numat tanh::activative(const numat& x)
		{
			// Activative the data
			fx = 2.0 / (1.0 + numat::exp(-_alpha * x)) - 1.0;

			return fx;
		}

		const numat tanh::derivative()
		{
			// Derivative the data
			df = ((_alpha / 2.0) * (1.0 + fx)).mul(1.0 - fx);

			return df;
		}

		relu::relu()
		{
			// Set an object
			setObject();
		}

		relu::relu(const relu& obj)
		{
			// Set an object
			setObject();

			// Clone Object
			*this = obj;
		}

		relu::~relu()
		{
			// Clear the object
			clearObject();
		}

		relu& relu::operator=(const relu& obj)
		{
			// Clear the object
			clearObject();

			// Set an object
			setObject();

			// Copy the object
			copyObject(obj);

			return *this;
		}

		void relu::setObject()
		{
			// Set the parameters
			setType(*this);
			_type = ACT_FUNC_ReLU;
			_alpha = 0.0;
		}

		const numat relu::activative(const numat& x)
		{
			// Activative the data
			t = x;
			fx = t;
			for (int i = 0; i < fx.length(); i++)
			{
				if (fx(i) < 0.0)
				{
					fx(i) = 0.0;
				}
			}

			return fx;
		}

		const numat relu::derivative()
		{
			// Derivative the data
			df = numat::ones(msize(fx.rows, fx.cols));
			for (int i = 0; i < df.length(); i++)
			{
				if (t(i) < 0.0)
				{
					df(i) = 0.0;
				}
			}

			return df;
		}

		relu6::relu6()
		{
			// Set an object
			setObject();
		}

		relu6::relu6(const relu6& obj)
		{
			// Set an object
			setObject();

			// Clone the object
			*this = obj;
		}

		relu6::~relu6()
		{
			// Clear the object
			clearObject();
		}

		relu6& relu6::operator=(const relu6& obj)
		{
			// Clear the object
			clearObject();

			// Set an object
			setObject();

			// Copy the object
			copyObject(obj);

			return *this;
		}

		void relu6::setObject()
		{
			// Set the parameters
			setType(*this);
			_type = ACT_FUNC_ReLU6;
			_alpha = 0.0;
		}

		const numat relu6::activative(const numat& x)
		{
			// Activative the data
			t = x;
			fx = t;
			for (int i = 0; i < fx.length(); i++)
			{
				if (fx(i) < 0.0)
				{
					fx(i) = 0.0;
				}
				else if (fx(i) > 6.0)
				{
					fx(i) = 6.0;
				}
			}

			return fx;
		}

		const numat relu6::derivative()
		{
			// Derivative the data
			df = numat::ones(msize(fx.rows, fx.cols));
			for (int i = 0; i < df.length(); i++)
			{
				if (t(i) < 0.0)
				{
					df(i) = 0.0;
				}
				else if (t(i) > 6.0)
				{
					df(i) = 0.0;
				}
			}

			return df;
		}

		leakyrelu::leakyrelu()
		{
			// Set an object
			setObject();
		}

		leakyrelu::leakyrelu(const leakyrelu& obj)
		{
			// Set an object
			setObject();

			// Clone the object
			*this = obj;
		}

		leakyrelu::~leakyrelu()
		{
			// Clear the object
			clearObject();
		}

		leakyrelu& leakyrelu::operator=(const leakyrelu& obj)
		{
			// Clear the object
			clearObject();

			// Set an object
			setObject();

			// Copy the object
			copyObject(obj);

			return *this;
		}

		void leakyrelu::setObject()
		{
			// Set the parameters
			setType(*this);
			_type = ACT_FUNC_LEAKY_ReLU;
			_alpha = 0.01;
		}

		const numat leakyrelu::activative(const numat& x)
		{
			// Activative the data
			t = x;
			fx = t;
			for (int i = 0; i < fx.length(); i++)
			{
				if (fx(i) < 0.0)
				{
					fx(i) = 0.01 * fx(i);
				}
			}

			return fx;
		}

		const numat leakyrelu::derivative()
		{
			// Derivative the data
			df = numat::ones(msize(fx.rows, fx.cols));
			for (int i = 0; i < df.length(); i++)
			{
				if (t(i) < 0.0)
				{
					df(i) = 0.01;
				}
			}

			return df;
		}

		prelu::prelu()
		{
			// Set an object
			setObject();
		}

		prelu::prelu(const double alpha)
		{
			// Set an object
			setObject();

			// Set the parameters
			set(alpha);
		}

		prelu::prelu(const prelu& obj)
		{
			// Set an object
			setObject();

			// Clone the object
			*this = obj;
		}

		prelu::~prelu()
		{
			// Clear the object
			clearObject();
		}

		prelu& prelu::operator=(const prelu& obj)
		{
			// Clear the object
			clearObject();

			// Set an object
			setObject();

			// Copy the object
			copyObject(obj);

			return *this;
		}

		void prelu::setObject()
		{
			// Set the parameters
			setType(*this);
			_type = ACT_FUNC_PReLU;
			_alpha = 0.0;
		}

		const numat prelu::activative(const numat& x)
		{
			// Activative the data
			t = x;
			fx = t;
			for (int i = 0; i < fx.length(); i++)
			{
				if (fx(i) < 0.0)
				{
					fx(i) = _alpha * fx(i);
				}
			}

			return fx;
		}

		const numat prelu::derivative()
		{
			// Derivative the data
			df = numat::ones(msize(fx.rows, fx.cols));
			for (int i = 0; i < df.length(); i++)
			{
				if (t(i) < 0.0)
				{
					df(i) = _alpha;
				}
			}

			return df;
		}

		elu::elu()
		{
			// Set an object
			setObject();
		}

		elu::elu(const double alpha)
		{
			// Set an object
			setObject();

			// Set the parameters
			set(alpha);
		}

		elu::elu(const elu& obj)
		{
			// Set an object
			setObject();

			// Clone the object
			*this = obj;
		}

		elu::~elu()
		{
			// Clear the object
			clearObject();
		}

		elu& elu::operator=(const elu& obj)
		{
			// Clear the object
			clearObject();

			// Set an object
			setObject();

			// Copy the object
			copyObject(obj);

			return *this;
		}

		void elu::setObject()
		{
			// Set the parameters
			setType(*this);
			_type = ACT_FUNC_ELU;
			_alpha = 1.0;
		}

		const numat elu::activative(const numat& x)
		{
			// Activative the data
			t = x;
			fx = t;
			for (int i = 0; i < fx.length(); i++)
			{
				if (fx(i) < 0.0)
				{
					fx(i) = _alpha * (exp(fx(i)) - 1.0);
				}
			}

			return fx;
		}

		const numat elu::derivative()
		{
			// Derivative the data
			df = numat::ones(msize(fx.rows, fx.cols));
			for (int i = 0; i < df.length(); i++)
			{
				if (t(i) < 0.0)
				{
					df(i) = fx(i) + _alpha;
				}
			}

			return df;
		}

		softsign::softsign()
		{
			// Set an object
			setObject();
		}

		softsign::softsign(const softsign& obj)
		{
			// Set an object
			setObject();

			// Clone the object
			*this = obj;
		}

		softsign::~softsign()
		{
			// Clear the object
			clearObject();
		}

		softsign& softsign::operator=(const softsign& obj)
		{
			// Clear the object
			clearObject();

			// Set an object
			setObject();

			// Copy the object
			copyObject(obj);

			return *this;
		}

		void softsign::setObject()
		{
			// Set the parameters
			setType(*this);
			_type = ACT_FUNC_SOFTSIGN;
			_alpha = 1.0;
		}

		const numat softsign::activative(const numat& x)
		{
			// Activative the data
			t = 1.0 + numat::abs(x);
			fx = x / t;

			return fx;
		}

		const numat softsign::derivative()
		{
			// Derivative the data
			df = 1.0 / (t.mul(t));

			return df;
		}

		softplus::softplus()
		{
			// Set an object
			setObject();
		}

		softplus::softplus(const softplus& obj)
		{
			// Set an object
			setObject();

			// Clone the object
			*this = obj;
		}

		softplus::~softplus()
		{
			// Clear the object
			clearObject();
		}

		softplus& softplus::operator=(const softplus& obj)
		{
			// Clear the object
			clearObject();

			// Set an object
			setObject();

			// Copy the object
			copyObject(obj);

			return *this;
		}

		void softplus::setObject()
		{
			// Set the parameters
			setType(*this);
			_type = ACT_FUNC_SOFTPLUS;
			_alpha = 1.0;
		}

		const numat softplus::activative(const numat& x)
		{
			// Activative the data
			t = x;
			fx = numat::log(1.0 + numat::exp(t));

			return fx;
		}

		const numat softplus::derivative()
		{
			// Derivative the data
			df = 1.0 / (1.0 + numat::exp(-t));

			return df;
		}

		softmax::softmax()
		{
			// Set an object
			setObject();
		}

		softmax::softmax(const softmax& obj)
		{
			// Set an object
			setObject();

			// Clone the object
			*this = obj;
		}

		softmax::~softmax()
		{
			// Clear the object
			clearObject();
		}

		softmax& softmax::operator=(const softmax& obj)
		{
			// Clear the object
			clearObject();

			// Set an object
			setObject();

			// Copy the object
			copyObject(obj);

			return *this;
		}

		void softmax::setObject()
		{
			// Set the parameters
			setType(*this);
			_type = ACT_FUNC_SOFTMAX;
			_alpha = 1.0;
		}

		const numat softmax::activative(const numat& x)
		{
			// Subtract maximum values
			numat m = numat::maxr(x);
			numat n(msize(x.rows, x.cols));
			for (int i = 0; i < x.rows; i++)
			{
				for (int j = 0; j < x.cols; j++)
				{
					n(i, j) = x(i, j) - m(i);			// overflow 방지
				}
			}

			// Activative the data
			numat e = numat::exp(n);
			numat s = numat::sumr(e);
			fx = numat(msize(e.rows, e.cols));
			for (int i = 0; i < e.rows; i++)
			{
				for (int j = 0; j < e.cols; j++)
				{
					fx(i, j) = e(i, j) / s(i);
				}
			}

			return fx;
		}

		const numat softmax::derivative()
		{
			// Do nothing
			return numat();
		}

		logsoftmax::logsoftmax()
		{
			// Set an object
			setObject();
		}

		logsoftmax::logsoftmax(const logsoftmax& obj)
		{
			// Set an object
			setObject();

			// Clone the object
			*this = obj;
		}

		logsoftmax::~logsoftmax()
		{
			// Clear the object
			clearObject();
		}

		logsoftmax& logsoftmax::operator=(const logsoftmax& obj)
		{
			// Clear the object
			clearObject();

			// Set an object
			setObject();

			// Copy the object
			copyObject(obj);

			return *this;
		}

		void logsoftmax::setObject()
		{
			// Set the parameters
			setType(*this);
			_type = ACT_FUNC_SOFTMAX;
			_alpha = 1.0;
		}

		const numat logsoftmax::activative(const numat& x)
		{
			// Subtract maximum values
			numat m = numat::maxr(x);
			numat n(msize(x.rows, x.cols));
			for (int i = 0; i < x.rows; i++)
			{
				for (int j = 0; j < x.cols; j++)
				{
					n(i, j) = x(i, j) - m(i);			// overflow 방지
				}
			}

			// Activative the data
			numat e = numat::exp(n);
			numat s = numat::log(numat::sumr(e));
			fx = numat(msize(e.rows, e.cols));
			for (int i = 0; i < e.rows; i++)
			{
				for (int j = 0; j < e.cols; j++)
				{
					fx(i, j) = n(i, j) - s(i);			// underflow 방지
				}
			}

			return fx;
		}

		const numat logsoftmax::derivative()
		{
			// Do nothing
			return numat();
		}
	}
}