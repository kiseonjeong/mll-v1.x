#include "stdafx.h"
#include "Optimizer.h"

namespace mll
{
	optimizer::optimizer() : type(_type), epsilon(_epsilon), delta(_delta), decay(_decay), beta1(_beta1), beta2(_beta2)
	{
		// Set an object
		setObject();
	}

	optimizer::~optimizer()
	{
		// Clear the object
		clearObject();
	}

	void optimizer::setObject()
	{
		// Set the parameters
		setType(*this);
		_type = OPTIMIZER_UNKNOWN;
		_epsilon = 0.001;
		_delta = 0.9;
		_decay = 0.99;
		_beta1 = 0.9;
		_beta2 = 0.999;
		atype = ANNEALING_UNKNOWN;
		cycle = 100;
		k = 1.0;
	}

	void optimizer::copyObject(const object& obj)
	{
		// Do down casting
		const optimizer* _obj = static_cast<const optimizer*>(&obj);

		// Copy the parameters
		_type = _obj->_type;
		_epsilon = _obj->_epsilon;
		_delta = _obj->_delta;
		_decay = _obj->_decay;
		_beta1 = _obj->_beta1;
		_beta2 = _obj->_beta2;
		atype = _obj->atype;
		cycle = _obj->cycle;
		k = _obj->k;
	}

	void optimizer::clearObject()
	{

	}

	void optimizer::set(const int cycle, const double k, const antype type)
	{
		// Set the parameters
		this->cycle = cycle;
		this->k = k;
		this->atype = type;
	}

	void optimizer::update(const int epoch)
	{
		// Check the annealer type
		if (this->type != ANNEALING_UNKNOWN)
		{
			// Check the epoch and the cycle
			if (epoch != 0 && epoch % cycle == 0)
			{
				// Calculate annealing
				switch (atype)
				{
				case ANNEALING_STEP: epsilon = epsilon * k; break;
				case ANNEALING_EXP: epsilon = epsilon * exp(-k * epoch); break;
				case ANNEALING_INV: epsilon = epsilon / (1.0 + k * epoch); break;
				}
			}
		}
	}

	sgd::sgd()
	{
		// Set an object
		setObject();
	}

	sgd::sgd(const double epsilon)
	{
		// Set an object
		setObject();

		// Set the parameters
		set(epsilon);
	}

	sgd::sgd(const sgd& obj)
	{
		// Set an object
		setObject();

		// Clone the object
		*this = obj;
	}

	sgd::~sgd()
	{
		// Clear the object
		clearObject();
	}

	sgd& sgd::operator=(const sgd& obj)
	{
		// Clear the object
		clearObject();

		// Set an object
		setObject();

		// Copy the object
		copyObject(obj);

		return *this;
	}

	void sgd::setObject()
	{
		// Set the parameters
		setType(*this);
		_type = OPTIMIZER_SGD;
		_epsilon = 0.01;
	}

	void sgd::set(const double epsilon)
	{
		// Set the parameters
		_epsilon = epsilon;
	}

	void sgd::set(const double epsilon, const double delta, const double decay, const double beta1, const double beta2)
	{
		// Set the parameters
		set(epsilon);
	}

	void sgd::create(const nml::numem<nml::numat>& W)
	{
		// Do nothing
	}

	const numat sgd::calculate(const numat& grad)
	{
		// Calculate a step size using the SGD
		return -_epsilon * grad;
	}

	const numat sgd::calculate(const int l, const double iter, const numat& grad)
	{
		// Calculate a step size using the SGD
		return calculate(grad);
	}

	momentum::momentum()
	{
		// Set an object
		setObject();
	}

	momentum::momentum(const double epsilon, const double delta)
	{
		// Set an object
		setObject();

		// Set the parameters
		set(epsilon, delta);
	}

	momentum::momentum(const momentum& obj)
	{
		// Set an object
		setObject();

		// Clone the object
		*this = obj;
	}

	momentum::~momentum()
	{
		// Clear the object
		clearObject();
	}

	momentum& momentum::operator=(const momentum& obj)
	{
		// Clear the object
		clearObject();

		// Set an object
		setObject();

		// Copy the object
		copyObject(obj);

		return *this;
	}

	void momentum::setObject()
	{
		// Set the parameters
		setType(*this);
		_type = OPTIMIZER_MOMENTUN;
		_epsilon = 0.01;
		_delta = 0.9;

		// Set the memories
		m.release();
	}

	void momentum::copyObject(const object& obj)
	{
		// Do down casting
		const momentum* _obj = static_cast<const momentum*>(&obj);

		// Copy the parameters
		_type = _obj->_type;
		_epsilon = _obj->_epsilon;
		_delta = _obj->_delta;

		// Copy the memories
		m = _obj->m;
	}

	void momentum::clearObject()
	{
		// Clear the memories
		m.release();
	}

	void momentum::set(const double epsilon, const double delta)
	{
		// Set the parameters
		_epsilon = epsilon;
		_delta = delta;
	}

	void momentum::set(const double epsilon, const double delta, const double decay, const double beta1, const double beta2)
	{
		// Set the parameters
		set(epsilon, delta);
	}

	void momentum::create(const numem<numat>& W)
	{
		// Check a cache memory
		if (m.empty() == true)
		{
			// Create the cache memory
			m = numem<numat>(msize(W.length()));

			// Initialize the cache memory
			for (int i = 0; i < m.length(); i++)
			{
				m(i) = numat::zeros(msize(W(i).rows, W(i).cols));
			}
		}
	}

	const numat momentum::calculate(const int l, const numat& grad)
	{
		// Calculate a step size using the Momentum
		m(l) = -_epsilon * grad + _delta * m(l);

		return m(l);
	}

	const numat momentum::calculate(const int l, const double iter, const numat& grad)
	{
		// Calculate a step size using the Momentum
		return calculate(l, grad);
	}

	nesterov::nesterov()
	{
		// Set an object
		setObject();
	}

	nesterov::nesterov(const double epsilon, const double delta)
	{
		// Set an object
		setObject();

		// Set the parameters
		set(epsilon, delta);
	}

	nesterov::nesterov(const nesterov& obj)
	{
		// Set an object
		setObject();

		// Clone the object
		*this = obj;
	}

	nesterov::~nesterov()
	{
		// Clear the object
		clearObject();
	}

	nesterov& nesterov::operator=(const nesterov& obj)
	{
		// Clear the object
		clearObject();

		// Set an object
		setObject();

		// Copy the object
		copyObject(obj);

		return *this;
	}

	void nesterov::setObject()
	{
		// Set the parameters
		setType(*this);
		_type = OPTIMIZER_NESTEROV;
		_epsilon = 0.01;
		_delta = 0.9;

		// Set the memories
		m.release();
	}

	void nesterov::copyObject(const object& obj)
	{
		// Do down casting
		const nesterov* _obj = static_cast<const nesterov*>(&obj);

		// Copy the parameters
		_type = _obj->_type;
		_epsilon = _obj->_epsilon;
		_delta = _obj->_delta;

		// Copy the memories
		m = _obj->m;
	}

	void nesterov::clearObject()
	{
		// Clear the memories
		m.release();
	}

	const numat nesterov::calculate(const int l, const nml::numat& grad)
	{
		// Calculate a step size using the NAG
		m(l) *= _delta;
		m(l) -= _epsilon * grad;

		return _delta * _delta * m(l) - (1.0 + _delta) * _epsilon * grad;
	}

	adagrad::adagrad()
	{
		// Set an object
		setObject();
	}

	adagrad::adagrad(const double epsilon)
	{
		// Set an object
		setObject();

		// Set the marameters
		set(epsilon);
	}

	adagrad::adagrad(const adagrad& obj)
	{
		// Set an object
		setObject();

		// Clone the object
		*this = obj;
	}

	adagrad::~adagrad()
	{
		// Clear the object
		clearObject();
	}

	adagrad& adagrad::operator=(const adagrad& obj)
	{
		// Clear the object
		clearObject();

		// Set an object
		setObject();

		// Copy the object
		copyObject(obj);

		return *this;
	}

	void adagrad::setObject()
	{
		// Set the parameters
		setType(*this);
		_type = OPTIMIZER_ADAGRAD;
		_epsilon = 0.01;

		// Set the memories
		h.release();
	}

	void adagrad::copyObject(const object& obj)
	{
		// Do down casting
		const adagrad* _obj = static_cast<const adagrad*>(&obj);

		// Copy the parameters
		_type = _obj->_type;
		_epsilon = _obj->_epsilon;

		// Copy the memories
		h = _obj->h;
	}

	void adagrad::clearObject()
	{
		// Clear the memories
		h.release();
	}

	void adagrad::set(const double epsilon)
	{
		// Set the parameters
		_epsilon = epsilon;
	}

	void adagrad::set(const double epsilon, const double delta, const double decay, const double beta1, const double beta2)
	{
		// Set the parameters
		set(epsilon);
	}

	void adagrad::create(const numem<numat>& W)
	{
		// Check a cache memory
		if (h.empty() == true)
		{
			// Create the cache memory
			h = numem<numat>(msize(W.length()));

			// Initialize the cache memory
			for (int i = 0; i < h.length(); i++)
			{
				h(i) = numat::zeros(msize(W(i).rows, W(i).cols));
			}
		}
	}

	const numat adagrad::calculate(const int l, const nml::numat& grad)
	{
		// Calculate a step size using the AdaGrad
		h(l) += grad.mul(grad);

		return -_epsilon * grad / numat::sqrt(h(l) + 1e-7);
	}

	const numat adagrad::calculate(const int l, const double iter, const numat& grad)
	{
		// Calculate a step size using the AdaGrad
		return calculate(l, grad);
	}

	rmsprop::rmsprop()
	{
		// Set an object
		setObject();
	}

	rmsprop::rmsprop(const double epsilon, const double decay)
	{
		// Set an object
		setObject();

		// Set the parameters
		set(epsilon, decay);
	}

	rmsprop::rmsprop(const rmsprop& obj)
	{
		// Set an object
		setObject();

		// Clone the object
		*this = obj;
	}

	rmsprop::~rmsprop()
	{
		// Clear the object
		clearObject();
	}

	rmsprop& rmsprop::operator=(const rmsprop& obj)
	{
		// Clear the object
		clearObject();

		// Set an object
		setObject();

		// Copy the object
		copyObject(obj);

		return *this;
	}

	void rmsprop::setObject()
	{
		// Set the parameters
		setType(*this);
		_type = OPTIMIZER_RMSPROP;
		_epsilon = 0.01;
		_decay = 0.99;

		// Set the memories
		h.release();
	}

	void rmsprop::copyObject(const object& obj)
	{
		// Do down casting
		const rmsprop* _obj = static_cast<const rmsprop*>(&obj);

		// Copy the parameters
		_type = _obj->_type;
		_epsilon = _obj->_epsilon;
		_decay = _obj->_decay;

		// Copy the memories
		h = _obj->h;
	}

	void rmsprop::set(const double epsilon, const double decay)
	{
		// Set the parameters
		_epsilon = epsilon;
		_decay = decay;
	}

	const numat rmsprop::calculate(const int l, const nml::numat& grad)
	{
		// Calculate a step size using the RMSProp
		h(l) *= _decay;
		h(l) += (1.0 - _decay) * grad.mul(grad);

		return -_epsilon * grad / numat::sqrt(h(l) + 1e-7);
	}

	adadelta::adadelta()
	{
		// Set an object
		setObject();
	}

	adadelta::adadelta(const double decay)
	{
		// Set an object
		setObject();

		// Set the parameters
		set(decay);
	}

	adadelta::adadelta(const adadelta& obj)
	{
		// Set an object
		setObject();

		// Clone the object
		*this = obj;
	}

	adadelta::~adadelta()
	{
		// Clear the object
		clearObject();
	}

	adadelta& adadelta::operator=(const adadelta& obj)
	{
		// Clear the object
		clearObject();

		// Set an object
		setObject();

		// Copy the object
		copyObject(obj);

		return *this;
	}

	void adadelta::setObject()
	{
		// Set the parameters
		setType(*this);
		_type = OPTIMIZER_ADADELTA;
		_decay = 0.95;

		// Set the memories
		h.release();
		s.release();
		step.release();
	}

	void adadelta::copyObject(const object& obj)
	{
		// Do down casting
		const adadelta* _obj = static_cast<const adadelta*>(&obj);

		// Copy the parameters
		_type = _obj->_type;
		_decay = _obj->_decay;

		// Copy the memories
		h = _obj->h;
		s = _obj->s;
		step = _obj->step;
	}

	void adadelta::clearObject()
	{
		// Clear the memories
		h.release();
		s.release();
		step.release();
	}

	void adadelta::set(const double decay)
	{
		// Set the parameters
		_decay = decay;
	}

	void adadelta::set(const double epsilon, const double delta, const double decay, const double beta1, const double beta2)
	{
		// Set the parameters
		set(decay);
	}

	const numat adadelta::calculate(const int l, const nml::numat& grad)
	{
		// Calculate a step size using the AdaDelta
		h(l) = _decay * h(l) + (1.0 - _decay) * (grad.mul(grad));
		step = -(numat::sqrt(s(l) + 1e-7) / numat::sqrt(h(l) + 1e-7)).mul(grad);
		s(l) = _decay * s(l) + (1.0 - _decay) * (step.mul(step));

		return step;
	}

	const numat adadelta::calculate(const int l, const double iter, const numat& grad)
	{
		// Calculate a step size using the AdaDelta
		return calculate(l, grad);
	}

	// Update the learning rate
	void adadelta::update(const int epoch)
	{
		// Check the annealer type
		if (this->type != ANNEALING_UNKNOWN)
		{
			// Check the epoch and the cycle
			if (epoch != 0 && epoch % cycle == 0)
			{
				// Calculate annealing
				for (int i = 0; i < s.length(); i++)
				{
					switch (atype)
					{
					case ANNEALING_STEP: s(i) = s(i) * k; break;
					case ANNEALING_EXP: s(i) = s(i) * exp(-k * epoch); break;
					case ANNEALING_INV: s(i) = s(i) / (1.0 + k * epoch); break;
					}
				}
			}
		}
	}

	void adadelta::create(const numem<numat>& W)
	{
		// Check the cache memories
		if (h.empty() == true)
		{
			// Create the cache memory
			h = numem<numat>(msize(W.length()));

			// Initialize the cache memory
			for (int i = 0; i < h.length(); i++)
			{
				h(i) = numat::zeros(msize(W(i).rows, W(i).cols));
			}
		}
		if (s.empty() == true)
		{
			// Create the cache memory
			s = numem<numat>(msize(W.length()));

			// Initialize the cache memory
			for (int i = 0; i < s.length(); i++)
			{
				s(i) = numat::zeros(msize(W(i).rows, W(i).cols));
			}
		}
	}

	adam::adam()
	{
		// Set an object
		setObject();
	}

	adam::adam(const double epsilon, const double beta1, const double beta2)
	{
		// Set an object
		setObject();

		// Set the parameters
		set(epsilon, beta1, beta2);
	}

	adam::adam(const adam& obj)
	{
		// Set an object
		setObject();

		// Clone the object
		*this = obj;
	}

	adam::~adam()
	{
		// Clear the object
		clearObject();
	}

	adam& adam::operator=(const adam& obj)
	{
		// Clear the object
		clearObject();

		// Set an object
		setObject();

		// Copy the object
		copyObject(obj);

		return *this;
	}

	void adam::setObject()
	{
		// Set the parameters
		setType(*this);
		_type = OPTIMIZER_ADAM;
		_epsilon = 0.001;
		_beta1 = 0.9;
		_beta2 = 0.999;

		// Set the memories
		v.release();
		h.release();
	}

	void adam::copyObject(const object& obj)
	{
		// Do down casting
		optimizer::copyObject(obj);
		const adam* _obj = static_cast<const adam*>(&obj);

		// Copy the parameters
		_type = _obj->_type;
		_epsilon = _obj->_epsilon;
		_beta1 = _obj->_beta1;
		_beta2 = _obj->_beta2;

		// Copy the memories
		v = _obj->v;
		h = _obj->h;
	}

	void adam::clearObject()
	{
		// Clear the memories
		v.release();
		h.release();
	}

	void adam::set(const double epsilon, const double beta1, const double beta2)
	{
		// Set the parameters
		_epsilon = epsilon;
		_beta1 = beta1;
		_beta2 = beta2;
	}

	void adam::set(const double epsilon, const double delta, const double decay, const double beta1, const double beta2)
	{
		// Set the parameters
		set(epsilon, beta1, beta2);
	}

	void adam::create(const numem<numat>& W)
	{
		// Check the cache memories
		if (v.empty() == true)
		{
			// Create the cache memory
			v = numem<numat>(msize(W.length()));

			// Initialize the cache memory
			for (int i = 0; i < v.length(); i++)
			{
				v(i) = numat::zeros(msize(W(i).rows, W(i).cols));
			}
		}
		if (h.empty() == true)
		{
			// Create the cache memory
			h = numem<numat>(msize(W.length()));

			// Initialize the cache memory
			for (int i = 0; i < h.length(); i++)
			{
				h(i) = numat::zeros(msize(W(i).rows, W(i).cols));
			}
		}
	}

	const numat adam::calculate(const int l, const double iter, const numat& grad)
	{
		// Calculate a step size using the Adam
		v(l) += (1.0 - _beta1) * (grad - v(l));
		h(l) += (1.0 - _beta2) * (grad.mul(grad) - h(l));

		return -_epsilon * sqrt(1.0 - pow(_beta2, iter + 1)) / (1.0 - pow(_beta1, iter + 1)) * v(l) / numat::sqrt(h(l) + 1e-7);
	}
}