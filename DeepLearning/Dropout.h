#ifndef DROPOUT_H
#define DROPOUT_H

namespace mll
{
	// The dropout layer type
	typedef enum _dotype
	{
		DROPOUT_UNKNOWN_LAYER = 0x00,
		DROPOUT_INPUT_LAYER,
		DROPOUT_HIDDEN_LAYER,
	} dotype;

	// The Dropout Regularizer
	class dropout : public nml::object
	{
		// Variables
	public:
		// Selection matrix
		nml::prop::get<dotype> type;
		// Dropout keep probability
		nml::prop::get<double> kprob;

		// Functions
	public:
		// Set parameters
		// kprob : keep probability value for the dropout
		// type : layer type
		void set(const double kprob, const dotype type = DROPOUT_HIDDEN_LAYER);
		// Generate dropout matrix
		void generate(const int length);
		// Activative the data
		const nml::numat forward(const nml::numat& net) const;
		// Derivative the data
		const nml::numat backward(const nml::numat& dout) const;

		// Operators
	public:
		dropout& operator=(const dropout& obj);

		// Constructors & Destructor
	public:
		dropout();
		dropout(const double kprob, const dotype type = DROPOUT_HIDDEN_LAYER);
		dropout(const dropout& obj);
		~dropout();

		// Variables
	private:
		// Selection matrix
		dotype _type;
		// Selection matrix
		nml::numat M;
		// Dropout keep probability
		double _kprob;

		// Functions
	private:
		// Set an object
		void setObject();
		// Copy the object
		void copyObject(const nml::object& obj);
		// Clear the object
		void clearObject();

	};
}

#endif