#ifndef OPTIMIZER_H
#define OPTIMIZER_H

namespace mll
{
	// The Optimizer Type
	typedef enum _optype
	{
		OPTIMIZER_UNKNOWN = -1,
		OPTIMIZER_SGD,
		OPTIMIZER_MOMENTUN,
		OPTIMIZER_NESTEROV,
		OPTIMIZER_ADAGRAD,
		OPTIMIZER_RMSPROP,
		OPTIMIZER_ADADELTA,
		OPTIMIZER_ADAM,
	} optype;

	// The Annealing Type
	typedef enum _antype
	{
		ANNEALING_UNKNOWN = -1,
		ANNEALING_STEP,
		ANNEALING_EXP,
		ANNEALING_INV,
	} antype;

	// The Optimizer for Neural Network
	class optimizer : public nml::object
	{
		// Variables
	public:
		// Optimizer type
		nml::prop::get<optype> type;
		// Learning rate
		nml::prop::all<double> epsilon;
		// Momentum rate
		nml::prop::all<double> delta;
		// Decay rate
		nml::prop::all<double> decay;
		// Beta 1 parameter for the adam optimizer
		nml::prop::all<double> beta1;
		// Beta 2 parameter for the adam optimizer
		nml::prop::all<double> beta2;

		// Functions
	public:
		// Set the optimizer parameters
		virtual void set(const double epsilon, const double delta, const double decay, const double beta1, const double beta2) = 0;
		// Create the cache memories
		virtual void create(const nml::numem<nml::numat>& W) = 0;
		// Calculate a step size
		virtual const nml::numat calculate(const int l, const double iter, const nml::numat& grad) = 0;
		// Set the annealing parameters
		// cycle : decay cycle
		// k : decay rate
		// STEP type, epsilon' = k * epsilon
		// EXP type, epsilon' = epsilon * exp(-k * epoch)
		// INV type, epsilon' = epsilon / (1 + k * epoch)
		virtual void set(const int cycle, const double k, const antype type);
		// Update the learning rate
		virtual void update(const int epoch);

		// Constructors & Destructor
	public:
		optimizer();
		virtual ~optimizer();

		// Variables
	protected:
		// Optimizer type
		optype _type;
		// Learning rate
		double _epsilon;
		// Momentum rate
		double _delta;
		// Decay rate
		double _decay;
		// Beta 1 parameter for the adam optimizer
		double _beta1;
		// Beta 2 parameter for the adam optimizer
		double _beta2;
		// Annealing type
		antype atype;
		// Annealing cycle
		int cycle;
		// Annealing parameter
		double k;

		// Functions
	protected:
		// Set an object
		virtual void setObject();
		// Copy the object
		virtual void copyObject(const nml::object& obj);
		// Clear the object
		virtual void clearObject();

	};

	// The SGD (Stochastic Gradient Descent) Optimizer
	class sgd : public optimizer
	{
		// Variables
	public:


		// Functions
	public:
		// Set the optimizer parameters
		// epsilon : learning rate
		void set(const double epsilon);
		// Calculate a step size
		const nml::numat calculate(const nml::numat& grad);

		// Operators
	public:
		sgd& operator=(const sgd& obj);

		// Constructors & Destructor
	public:
		sgd();
		// epsilon : learning rate
		sgd(const double epsilon);
		sgd(const sgd& obj);
		~sgd();

		// Variables
	private:


		// Functions
	private:
		// Set an object
		void setObject();
		// Set the optimizer parameters
		void set(const double epsilon, const double delta, const double decay, const double beta1, const double beta2);
		// Create the cache memories
		void create(const nml::numem<nml::numat>& W);
		// Calculate a step size
		const nml::numat calculate(const int l, const double iter, const nml::numat& grad);

	};

	// The Momentum Optimizer
	class momentum : public optimizer
	{
		// Variables
	public:


		// Functions
	public:
		// Set the optimizer parameters
		// epsilon : learning rate
		// delta : momentum rate (default = 0.9)
		void set(const double epsilon, const double delta = 0.9);
		// Create the cache memories
		virtual void create(const nml::numem<nml::numat>& W);
		// Calculate a step size
		virtual const nml::numat calculate(const int l, const nml::numat& grad);

		// Operators
	public:
		momentum& operator=(const momentum& obj);

		// Constructors & Destructor
	public:
		momentum();
		// epsilon : learning rate
		// delta : momentum rate (default = 0.9)
		momentum(const double epsilon, const double delta = 0.9);
		momentum(const momentum& obj);
		virtual ~momentum();

		// Variables
	protected:
		// Cache Memory
		nml::numem<nml::numat> m;

		// Functions
	protected:
		// Set an object
		virtual void setObject();
		// Copy the object
		virtual void copyObject(const nml::object& obj);
		// Clear the object
		virtual void clearObject();
		// Set the optimizer parameters
		void set(const double epsilon, const double delta, const double decay, const double beta1, const double beta2);
		// Calculate a step size
		virtual const nml::numat calculate(const int l, const double iter, const nml::numat& grad);

	};

	// The NAG (Nesterov Accelerated Gradient) Optimizer
	class nesterov : public momentum
	{
		// Variables
	public:


		// Functions
	public:
		// Calculate a step size
		const nml::numat calculate(const int l, const nml::numat& grad);

		// Operators
	public:
		nesterov& operator=(const nesterov& obj);

		// Constructors & Destructor
	public:
		nesterov();
		// epsilon : learning rate
		// delta : momentum rate (default = 0.9)
		nesterov(const double epsilon, const double delta);
		nesterov(const nesterov& obj);
		~nesterov();

		// Variables
	private:


		// Functions
	private:
		// Set an object
		void setObject();
		// Copy the object
		void copyObject(const nml::object& obj);
		// Clear the object
		void clearObject();

	};

	// The AdaGrad Optimizer
	class adagrad : public optimizer
	{
		// Variables
	public:


		// Functions
	public:
		// Set the optimizer parameters
		// epsilon : learning rate (default = 0.01)
		void set(const double epsilon = 0.01);
		// Create the cache memories
		virtual void create(const nml::numem<nml::numat>& W);
		// Calculate a step size
		virtual const nml::numat calculate(const int l, const nml::numat& grad);

		// Operators
	public:
		adagrad& operator=(const adagrad& obj);

		// Constructors & Destructor
	public:
		adagrad();
		// epsilon : learning rate (default = 0.01)
		adagrad(const double epsilon);
		adagrad(const adagrad& obj);
		virtual ~adagrad();

		// Variables
	protected:
		// Cache Memory
		nml::numem<nml::numat> h;

		// Functions
	protected:
		// Set an object
		virtual void setObject();
		// Copy the object
		virtual void copyObject(const nml::object& obj);
		// Clear the object
		void clearObject();
		// Set the optimizer parameters
		void set(const double epsilon, const double delta, const double decay, const double beta1, const double beta2);
		// Calculate a step size
		const nml::numat calculate(const int l, const double iter, const nml::numat& grad);

	};

	// The RMSProp Optimizer
	class rmsprop : public adagrad
	{
		// Variables
	public:


		// Functions
	public:
		// Set the optimizer parameters
		// epsilon : learning rate (default = 0.01)
		// decay : decay rate (default = 0.99)
		void set(const double epsilon = 0.01, const double decay = 0.99);
		// Calculate a step size
		const nml::numat calculate(const int l, const nml::numat& grad);

		// Operators
	public:
		rmsprop& operator=(const rmsprop& obj);

		// Constructors & Destructor
	public:
		rmsprop();
		// epsilon : learning rate (default = 0.01)
		// decay : decay rate (default = 0.99)
		rmsprop(const double epsilon, const double decay);
		rmsprop(const rmsprop& obj);
		~rmsprop();

		// Variables
	private:


		// Functions
	private:
		// Set an object
		void setObject();
		// Copy the object
		void copyObject(const nml::object& obj);

	};

	// The AdaDelta Optimizer
	class adadelta : public optimizer
	{
		// Variables
	public:


		// Functions
	public:
		// Set the optimizer parameters
		// decay : decay rate (default = 0.95)
		void set(const double decay = 0.95);
		// Create the cache memories
		void create(const nml::numem<nml::numat>& W);
		// Calculate a step size
		const nml::numat calculate(const int l, const nml::numat& grad);
		// Update the learning rate
		void update(const int epoch);

		// Operators
	public:
		adadelta& operator=(const adadelta& obj);

		// Constructors & Destructor
	public:
		adadelta();
		// decay : decay rate (default = 0.95)
		adadelta(const double decay);
		adadelta(const adadelta& obj);
		~adadelta();

		// Variables
	private:
		// Decay Rate
		double _decay;
		// Cache Memory
		nml::numem<nml::numat> h;
		nml::numem<nml::numat> s;
		nml::numat step;

		// Functions
	private:
		// Set an object
		void setObject();
		// Copy the object
		void copyObject(const nml::object& obj);
		// Clear the object
		void clearObject();
		// Set the optimizer parameters
		void set(const double epsilon, const double delta, const double decay, const double beta1, const double beta2);
		// Calculate a step size
		const nml::numat calculate(const int l, const double iter, const nml::numat& grad);

	};

	// The Adam Optimizer
	class adam : public optimizer
	{
		// Variables
	public:


		// Functions
	public:
		// Set the optimizer parameters
		// epsilon : learning rate (default = 0.001)
		// beta1 : beta1 paramter for the Adam optimizer (default = 0.9)
		// beta2 : beta2 paramter for the Adam optimizer (default = 0.999)
		void set(const double epsilon = 0.001, const double beta1 = 0.9, const double beta2 = 0.999);
		// Create the cache memories
		void create(const nml::numem<nml::numat>& W);
		// Calculate a step size
		const nml::numat calculate(const int l, const double iter, const nml::numat& grad);

		// Operators
	public:
		adam& operator=(const adam& obj);

		// Constructors & Destructor
	public:
		adam();
		// epsilon : learning rate (default = 0.001)
		// beta1 : beta1 paramter for the Adam optimizer (default = 0.9)
		// beta2 : beta2 paramter for the Adam optimizer (default = 0.999)
		adam(const double epsilon, const double beta1, const double beta2);
		adam(const adam& obj);
		~adam();

		// Variables
	private:
		// Cache Memory
		nml::numem<nml::numat> v;
		nml::numem<nml::numat> h;

		// Functions
	private:
		// Set an object
		void setObject();
		// Copy the object
		void copyObject(const nml::object& obj);
		// Clear the object
		void clearObject();
		// Set the optimizer parameters
		void set(const double epsilon, const double delta, const double decay, const double beta1, const double beta2);

	};
}

#endif