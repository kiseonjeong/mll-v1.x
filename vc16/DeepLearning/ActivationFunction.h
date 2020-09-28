#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

namespace mll
{
	// For the Neural Network
	namespace nn
	{
		// The Activation Function Type
		typedef enum _aftype
		{
			ACT_FUNC_UNKNOWN = 0,
			ACT_FUNC_IDENTITY,
			ACT_FUNC_LINEAR,
			ACT_FUNC_SIGMOID,
			ACT_FUNC_HYPER_TANGENT,
			ACT_FUNC_ReLU,
			ACT_FUNC_ReLU6,
			ACT_FUNC_LEAKY_ReLU,
			ACT_FUNC_PReLU,
			ACT_FUNC_ELU,
			ACT_FUNC_SOFTSIGN,
			ACT_FUNC_SOFTPLUS,
			ACT_FUNC_SOFTMAX,
		} aftype;

		// The Activation Function for Neural Network
		class actfunc : public nml::object
		{
			// Variables
		public:
			// Activation function type
			nml::prop::get<aftype> type;
			// The alpha hyperparameter for the activation function
			nml::prop::get<double> alpha;

			// Functions
		public:
			// Set a hyperparameter
			virtual void set(const double alpha) = 0;
			// Calculate the activative data
			virtual const nml::numat activative(const nml::numat& x) = 0;
			// Calculate the derivative data
			virtual const nml::numat derivative() = 0;

			// Constructors & Destructor
		public:
			actfunc();
			virtual ~actfunc();

			// Variables
		protected:
			// Activation function type
			aftype _type;
			// The alpha hyperparameter for the activation function
			double _alpha;

			// Functions
		protected:
			// Set an object
			virtual void setObject();
			// Copy the object
			virtual void copyObject(const nml::object& obj);
			// Clear the object
			virtual void clearObject();

		};

		// The Identity Activation Function
		class identity : public actfunc
		{
			// Variables
		public:


			// Functions
		public:
			// Set a hyperparameter
			virtual void set(const double alpha);
			// Calculate the activative data
			virtual const nml::numat activative(const nml::numat& x);
			// Calculate the derivative data
			virtual const nml::numat derivative();

			// Operators
		public:
			identity& operator=(const identity& obj);

			// Constructors & Destructor
		public:
			identity();
			identity(const identity& obj);
			virtual ~identity();

			// Variables
		protected:
			// The activative data
			nml::numat fx;
			// The derivative data
			nml::numat df;
			// The temporary data
			nml::numat t;

			// Functions
		protected:
			// Set an object
			virtual void setObject();
			// Copy the object
			virtual void copyObject(const nml::object& obj);
			// Clear the object
			virtual void clearObject();

		};

		// The Linear Activation Function
		class linear : public identity
		{
			// Variables
		public:


			// Functions
		public:
			// Calculate the activative data
			const nml::numat activative(const nml::numat& x);
			// Calculate the derivative data
			const nml::numat derivative();

			// Operators
		public:
			linear& operator=(const linear& obj);

			// Constructors & Destructor
		public:
			linear();
			linear(const double alpha);
			linear(const linear& obj);
			~linear();

			// Variables
		private:


			// Functions
		private:
			// Set an object
			void setObject();

		};

		// The Sigmoid Activation Function
		class sigmoid : public identity
		{
			// Variables
		public:


			// Functions
		public:
			// Activative the data
			const nml::numat activative(const nml::numat& x);
			// Derivative the data
			const nml::numat derivative();

			// Operators
		public:
			sigmoid& operator=(const sigmoid& obj);

			// Constructors & Destructor
		public:
			sigmoid();
			sigmoid(const double alpha);
			sigmoid(const sigmoid& obj);
			~sigmoid();

			// Variables
		private:


			// Functions
		private:
			// Set an object
			void setObject();

		};

		// The Hyperbolic Tangent Activation Function
		class tanh : public identity
		{
			// Variables
		public:


			// Functions
		public:
			// Activative the data
			const nml::numat activative(const nml::numat& x);
			// Derivative the data
			const nml::numat derivative();

			// Operators
		public:
			tanh& operator=(const tanh& obj);

			// Constructors & Destructor
		public:
			tanh();
			tanh(const double alpha);
			tanh(const tanh& obj);
			~tanh();

			// Variables
		private:


			// Functions
		private:
			// Set an object
			void setObject();

		};

		// The Rectified Linear Unit Activation Function
		class relu : public identity
		{
			// Variables
		public:


			// Functions
		public:
			// Activative the data
			const nml::numat activative(const nml::numat& x);
			// Derivative the data
			const nml::numat derivative();

			// Operators
		public:
			relu& operator=(const relu& obj);

			// Constructors & Destructor
		public:
			relu();
			relu(const relu& obj);
			~relu();

			// Variables
		private:


			// Functions
		private:
			// Set an object
			void setObject();

		};

		// The Rectified Linear Unit6 Activation Function
		class relu6 : public identity
		{
			// Variables
		public:


			// Functions
		public:
			// Activative the data
			const nml::numat activative(const nml::numat& x);
			// Derivative the data
			const nml::numat derivative();

			// Operators
		public:
			relu6& operator=(const relu6& obj);

			// Constructors & Destructor
		public:
			relu6();
			relu6(const relu6& obj);
			~relu6();

			// Variables
		private:


			// Functions
		private:
			// Set an object
			void setObject();

		};

		// The Leaky Rectified Linear Unit Activation Function
		class leakyrelu : public identity
		{
			// Variables
		public:


			// Functions
		public:
			// Activative the data
			const nml::numat activative(const nml::numat& x);
			// Derivative the data
			const nml::numat derivative();

			// Operators
		public:
			leakyrelu& operator=(const leakyrelu& obj);

			// Constructors & Destructor
		public:
			leakyrelu();
			leakyrelu(const leakyrelu& obj);
			~leakyrelu();

			// Variables
		private:


			// Functions
		private:
			// Set an object
			void setObject();

		};

		// The Parameteric Rectified Linear Unit Activation Function
		class prelu : public identity
		{
			// Variables
		public:


			// Functions
		public:
			// Activative the data
			const nml::numat activative(const nml::numat& x);
			// Derivative the data
			const nml::numat derivative();

			// Operators
		public:
			prelu& operator=(const prelu& obj);

			// Constructors & Destructor
		public:
			prelu();
			prelu(const double alpha);
			prelu(const prelu& obj);
			~prelu();

			// Variables
		private:


			// Functions
		private:
			// Set an object
			void setObject();

		};

		// The Exponential Linear Unit Activation Function
		class elu : public identity
		{
			// Variables
		public:


			// Functions
		public:
			// Activative the data
			const nml::numat activative(const nml::numat& x);
			// Derivative the data
			const nml::numat derivative();

			// Operators
		public:
			elu& operator=(const elu& obj);

			// Constructors & Destructor
		public:
			elu();
			elu(const double alpha);
			elu(const elu& obj);
			~elu();

			// Variables
		private:


			// Functions
		private:
			// Set an object
			void setObject();

		};

		// The Soft Sign Activation Function
		class softsign : public identity
		{
			// Variables
		public:


			// Functions
		public:
			// Activative the data
			const nml::numat activative(const nml::numat& x);
			// Derivative the data
			const nml::numat derivative();

			// Operators
		public:
			softsign& operator=(const softsign& obj);

			// Constructors & Destructor
		public:
			softsign();
			softsign(const softsign& obj);
			virtual ~softsign();

			// Variables
		private:


			// Functions
		private:
			// Set an object
			void setObject();

		};

		// The Soft Plus Activation Function
		class softplus : public identity
		{
			// Variables
		public:


			// Functions
		public:
			// Activative the data
			const nml::numat activative(const nml::numat& x);
			// Derivative the data
			const nml::numat derivative();

			// Operators
		public:
			softplus& operator=(const softplus& obj);

			// Constructors & Destructor
		public:
			softplus();
			softplus(const softplus& obj);
			~softplus();

			// Variables
		private:


			// Functions
		private:
			// Set an object
			void setObject();

		};

		// The Softmax Activation Function
		class softmax : public identity
		{
			// Variables
		public:


			// Functions
		public:
			// Activative the data
			const nml::numat activative(const nml::numat& x);
			// Derivative the data
			const nml::numat derivative();

			// Operators
		public:
			softmax& operator=(const softmax& obj);

			// Constructors & Destructor
		public:
			softmax();
			softmax(const softmax& obj);
			~softmax();

			// Variables
		private:


			// Functions
		private:
			// Set an object
			void setObject();

		};

		// The Log Softmax Activation Function
		class logsoftmax : public identity
		{
			// Variables
		public:


			// Functions
		public:
			// Activative the data
			const nml::numat activative(const nml::numat& x);
			// Derivative the data
			const nml::numat derivative();

			// Operators
		public:
			logsoftmax& operator=(const logsoftmax& obj);

			// Constructors & Destructor
		public:
			logsoftmax();
			logsoftmax(const logsoftmax& obj);
			~logsoftmax();

			// Variables
		private:


			// Functions
		private:
			// Set an object
			void setObject();

		};
	}
}

#endif