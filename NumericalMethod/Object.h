#ifndef OBJECT_H
#define OBJECT_H

namespace nml
{
	class object
	{
		// Variables
	public:


		// Functions
	public:
		// Get the object type
		const std::string getType() const;
		// Compare the objects
		template<typename T> bool isEqualType(const T& var) const;

		// Operators
	public:
		object& operator=(const object& obj);

		// Constructors & Destructor
	public:
		object();
		object(const object& obj);
		virtual ~object();

		// Variables
	protected:
		// Object type
		std::string objType;

		// Functions
	protected:
		// Set an object
		virtual void setObject();
		// Copy the object
		virtual void copyObject(const object& obj);
		// Clear the object
		virtual void clearObject();
		// Set a object type
		template<typename T> void setType(const T& var);

	};
}

#include "Object.hpp"

#endif