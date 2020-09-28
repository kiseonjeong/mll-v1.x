#ifndef ANNEALING_H
#define ANNEALING_H

namespace mll
{
	// The Annealing Type
	typedef enum _antype
	{
		ANNEALING_UNKNOWN = -1,
		ANNEALING_STEP,
		ANNEALING_EXP,
		ANNEALING_INV,
	} antype;

	// The Annealing for Neural Network
	class annealing : public nml::object
	{
		// Variables
	public:


		// Functions
	public:
		// Set the parameters
		// cycle : decay cycle
		// k : decay rate
		// STEP type, epsilon' = k * epsilon
		// EXP type, epsilon' = epsilon * exp(-k * epoch)
		// INV type, epsilon' = epsilon / (1 + k * epoch)
		void set(const int cycle, const double k, const antype type);
		// Update the learning rate
		void update(const int epoch, optimizer* opt) const;

		// Operators
	public:
		annealing& operator=(const annealing& obj);

		// Constructors & Destructor
	public:
		annealing();
		// cycle : decay cycle
		// k : decay rate
		// STEP type, epsilon' = k * epsilon
		// EXP type, epsilon' = epsilon * exp(-k * epoch)
		// INV type, epsilon' = epsilon / (1 + k * epoch)
		annealing(const int cycle, const double k, const antype type);
		annealing(const annealing& obj);
		~annealing();

		// Variables
	private:
		// Annealing type
		antype type;
		// Annealing cycle
		int cycle;
		// Annealing parameter
		double k;

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