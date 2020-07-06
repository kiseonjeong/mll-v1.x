#include "stdafx.h"
#include "NetworkLayer.h"

namespace mll
{
	netlayer::netlayer() : type(_type), node(_node)
	{
		// Set an object
		setObject();
	}

	netlayer::netlayer(const int node) : type(_type), node(_node)
	{
		// Set an object
		setObject();

		// Set the number of the nodes
		_node = node;
	}

	netlayer::netlayer(const int node, const nn::actfunc& afunc) : type(_type), node(_node)
	{
		// Set an object
		setObject();

		// Create a network layer
		create(node, afunc);
	}

	netlayer::netlayer(const netlayer& obj) : type(_type), node(_node)
	{
		// Set an object
		setObject();

		// Clone the object
		*this = obj;
	}

	netlayer::~netlayer()
	{
		// Clear the object
		clearObject();
	}

	netlayer& netlayer::operator=(const netlayer& obj)
	{
		// Clear the object
		clearObject();

		// Set an object
		setObject();

		// Copy the object
		copyObject(obj);

		return *this;
	}

	void netlayer::setObject()
	{
		// Set the parameters
		setType(*this);
		_node = 0;
		set(NET_LAYER_HIDDEN);

		// Set the memories
		afunc = nullptr;
	}

	void netlayer::copyObject(const object& obj)
	{
		// Do down casting
		const netlayer* _obj = static_cast<const netlayer*>(&obj);

		// Copy the parameters
		_type = _obj->_type;
		_node = _obj->_node;

		// Copy the memories
		if (_obj->afunc != nullptr)
		{
			createActFunc(*_obj->afunc);
		}
	}

	void netlayer::clearObject()
	{
		// Clear the memories
		if (afunc != nullptr)
		{
			delete afunc;
		}
	}

	void netlayer::create(const int node, const nn::actfunc& afunc)
	{
		// Set the number of the nodes
		_node = node;

		// Set the activation function
		createActFunc(afunc);
	}

	void netlayer::set(const nn::actfunc& afunc)
	{
		// Set an activation function
		createActFunc(afunc);
	}

	void netlayer::set(const nltype type)
	{
		// Set a layer type
		_type = type;
	}

	void netlayer::createActFunc(const nn::actfunc& afunc)
	{
		// Check the old memory
		if (this->afunc != nullptr)
		{
			delete this->afunc;
		}

		// Create an activation function
		switch (afunc.type)
		{
		case nn::ACT_FUNC_IDENTITY: this->afunc = new nn::identity((nn::identity&)afunc); break;
		case nn::ACT_FUNC_LINEAR: this->afunc = new nn::linear((nn::linear&)afunc); break;
		case nn::ACT_FUNC_SIGMOID: this->afunc = new nn::sigmoid((nn::sigmoid&)afunc); break;
		case nn::ACT_FUNC_HYPER_TANGENT: this->afunc = new nn::tanh((nn::tanh&)afunc); break;
		case nn::ACT_FUNC_ReLU: this->afunc = new nn::relu((nn::relu&)afunc); break;
		case nn::ACT_FUNC_ReLU6: this->afunc = new nn::relu6((nn::relu6&)afunc); break;
		case nn::ACT_FUNC_LEAKY_ReLU: this->afunc = new nn::leakyrelu((nn::leakyrelu&)afunc); break;
		case nn::ACT_FUNC_PReLU: this->afunc = new nn::prelu((nn::prelu&)afunc); break;
		case nn::ACT_FUNC_ELU: this->afunc = new nn::elu((nn::elu&)afunc); break;
		case nn::ACT_FUNC_SOFTSIGN: this->afunc = new nn::softsign((nn::softsign&)afunc); break;
		case nn::ACT_FUNC_SOFTPLUS: this->afunc = new nn::softplus((nn::softplus&)afunc); break;
// 		case nn::ACT_FUNC_SOFTMAX: this->afunc = new nn::softmax((nn::softmax&)afunc); break;
		case nn::ACT_FUNC_SOFTMAX: this->afunc = new nn::logsoftmax((nn::logsoftmax&)afunc); break;
		default: this->afunc = nullptr; break;
		}
	}

	const numat netlayer::activative(const numat& x) const
	{
		// Check the function
		if (afunc != nullptr)
		{
			// Activate the function
			return afunc->activative(x);
		}
		else
		{
			return numat();
		}
	}

	const numat netlayer::derivative() const
	{
		// Check the function
		if (afunc != nullptr)
		{
			// Derivative the function
			return afunc->derivative();
		}
		else
		{
			return numat();
		}
	}

	const nn::actfunc& netlayer::getActFunc() const
	{
		// Get the activation function
		return *afunc;
	}
}