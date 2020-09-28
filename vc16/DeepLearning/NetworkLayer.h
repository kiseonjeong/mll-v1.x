#ifndef LAYER_H
#define LAYER_H

namespace mll
{
	// The Network Layer Type for Multi Layer Perceptron
	typedef enum _nltype
	{
		NET_LAYER_UNKNOWN = -1,
		NET_LAYER_INPUT,
		NET_LAYER_HIDDEN,
		NET_LAYER_OUTPUT,
	} nltype;

	// The Network Layer for Neural Network
	class netlayer : public nml::object
	{
		// Variables
	public:
		// The number of types
		nml::prop::get<nltype> type;
		// The number of nodes
		nml::prop::get<int> node;

		// Functions
	public:
		// Create a layer architecture
		void create(const int node, const nn::actfunc& afunc);
		// Set an activation function
		void set(const nn::actfunc& afunc);
		// Set a network layer type
		void set(const nltype type);
		// Activative the data
		const nml::numat activative(const nml::numat& x) const;
		// Derivative the data
		const nml::numat derivative() const;
		// Get the activation function
		const nn::actfunc& getActFunc() const;

		// Operators
	public:
		netlayer& operator=(const netlayer& obj);

		// Constructors & Destructor
	public:
		netlayer();
		netlayer(const int node);
		netlayer(const int node, const nn::actfunc& afunc);
		netlayer(const netlayer& obj);
		~netlayer();

		// Variables
	private:
		// Network layer type
		nltype _type;
		// Activation function
		nn::actfunc* afunc;
		// The number of nodes
		int _node;

		// Functions
	private:
		// Set an object
		void setObject();
		// Copy the object
		void copyObject(const nml::object& obj);
		// Clear the object
		void clearObject();
		// Create the activation function
		void createActFunc(const nn::actfunc& afunc);

	};
}

#endif