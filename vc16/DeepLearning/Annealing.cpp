#include "stdafx.h"
#include "Annealing.h"

namespace mll
{
	annealing::annealing()
	{
		// Set an object
		setObject();
	}

	annealing::annealing(const int cycle, const double k, const antype type)
	{
		// Set an object
		setObject();

		// Set the parameters
		set(cycle, k, type);
	}

	annealing::annealing(const annealing& obj)
	{
		// Set an object
		setObject();

		// Clone the object
		*this = obj;
	}

	annealing::~annealing()
	{
		// Clear the object
		clearObject();
	}

	annealing& annealing::operator=(const annealing& obj)
	{
		// Clear the object
		clearObject();

		// Set an object
		setObject();

		// Copy the object
		copyObject(obj);

		return *this;
	}

	void annealing::setObject()
	{
		// Set the parameters
		setType(*this);
		type = ANNEALING_UNKNOWN;
		cycle = 100;
		k = 1.0;
	}

	void annealing::copyObject(const object& obj)
	{
		// Do down casting
		annealing* _obj = (annealing*)&obj;

		// Copy the parameters
		type = _obj->type;
		cycle = _obj->cycle;
		k = _obj->k;
	}

	void annealing::clearObject()
	{

	}

	void annealing::set(const int cycle, const double k, const antype type)
	{
		// Set the parameters
		this->cycle = cycle;
		this->k = k;
		this->type = type;
	}

	void annealing::update(const int epoch, optimizer* opt) const
	{
		// Check the annealer type
		if (this->type != ANNEALING_UNKNOWN)
		{
			// Check the epoch and the cycle
			if (epoch != 0 && epoch % cycle == 0)
			{
				// Check the optimizer type
				if (opt->type == OPTIMIZER_ADADELTA)
				{
					// Do down casting
					adadelta* _opt = (adadelta*)opt;

					// Calculate annealing
					numem<numat> epsilon = _opt->s;
					for (int i = 0; i < epsilon.length(); i++)
					{
						switch (type)
						{
						case ANNEALING_STEP: epsilon(i) = epsilon(i) * k; break;
						case ANNEALING_EXP: epsilon(i) = epsilon(i) * exp(-k * epoch); break;
						case ANNEALING_INV: epsilon(i) = epsilon(i) / (1.0 + k * epoch); break;
						}
					}

					// Save the learning rate
					_opt->s = epsilon;
				}
				else
				{
					// Calculate annealing
					switch (type)
					{
					case ANNEALING_STEP: opt->epsilon = opt->epsilon * k; break;
					case ANNEALING_EXP: opt->epsilon = opt->epsilon * exp(-k * epoch); break;
					case ANNEALING_INV: opt->epsilon = opt->epsilon / (1.0 + k * epoch); break;
					}
				}
			}
		}
	}
}