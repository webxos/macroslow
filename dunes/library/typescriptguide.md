# TypeScript Mastery Guide: From Basics to Advanced

## Table of Contents
1. [Introduction to TypeScript](#introduction)
2. [Basic Types](#basic-types)
3. [Interfaces and Type Aliases](#interfaces-types)
4. [Functions](#functions)
5. [Classes and OOP](#classes-oop)
6. [Generics](#generics)
7. [Advanced Types](#advanced-types)
8. [Modules and Namespaces](#modules-namespaces)
9. [Decorators](#decorators)
10. [Utility Types](#utility-types)
11. [TypeScript with React](#typescript-react)
12. [Project Configuration](#project-configuration)
13. [Best Practices](#best-practices)

## 1. Introduction to TypeScript <a name="introduction"></a>

TypeScript is a superset of JavaScript that adds static typing and other features to help build large-scale applications.

### Why TypeScript?
- **Type Safety**: Catch errors at compile time
- **Better Tooling**: Enhanced autocomplete and refactoring
- **Readable Code**: Self-documenting code with types
- **ECMAScript Support**: Use latest JavaScript features

### Installation
```bash
npm install -g typescript
# or
yarn global add typescript
```

### Basic Compilation
```bash
tsc filename.ts  # Compiles to filename.js
tsc --watch      # Watch mode for development
```

## 2. Basic Types <a name="basic-types"></a>

### Primitive Types
```typescript
// String
let name: string = "John";
let message: string = `Hello, ${name}!`;

// Number
let age: number = 30;
let price: number = 19.99;

// Boolean
let isActive: boolean = true;

// Array
let numbers: number[] = [1, 2, 3];
let names: Array<string> = ["John", "Jane"];

// Tuple
let person: [string, number] = ["John", 30];

// Enum
enum Color {
  Red,
  Green,
  Blue
}
let favoriteColor: Color = Color.Blue;

// Any (use sparingly)
let dynamicValue: any = "could be anything";
dynamicValue = 42;

// Void (for functions that don't return)
function logMessage(): void {
  console.log("Hello");
}

// Null and Undefined
let nullValue: null = null;
let undefinedValue: undefined = undefined;

// Never (for functions that never return)
function throwError(message: string): never {
  throw new Error(message);
}

// Unknown (safer alternative to any)
let unknownValue: unknown = "Hello";
if (typeof unknownValue === "string") {
  let stringValue: string = unknownValue;
}
```

### Type Assertions
```typescript
let someValue: any = "this is a string";
let strLength: number = (someValue as string).length;
// or
let strLength2: number = (<string>someValue).length;
```

## 3. Interfaces and Type Aliases <a name="interfaces-types"></a>

### Interfaces
```typescript
interface User {
  id: number;
  name: string;
  email: string;
  age?: number; // Optional property
  readonly createdAt: Date; // Readonly property
}

function createUser(user: User): User {
  return {
    ...user,
    createdAt: new Date()
  };
}

const newUser: User = createUser({
  id: 1,
  name: "John Doe",
  email: "john@example.com"
});

// Interface with function
interface MathOperation {
  (x: number, y: number): number;
}

const add: MathOperation = (a, b) => a + b;

// Extending interfaces
interface Admin extends User {
  permissions: string[];
}

const admin: Admin = {
  id: 1,
  name: "Admin",
  email: "admin@example.com",
  createdAt: new Date(),
  permissions: ["read", "write", "delete"]
};

// Index signatures
interface StringArray {
  [index: number]: string;
}

const myArray: StringArray = ["a", "b", "c"];
```

### Type Aliases
```typescript
type ID = number | string;
type UserRole = "admin" | "user" | "guest";

type Point = {
  x: number;
  y: number;
  z?: number;
};

type UserProfile = User & {
  avatar: string;
  preferences: string[];
};

// Union types
type Status = "success" | "error" | "loading";

// Intersection types
type Employee = User & {
  employeeId: number;
  department: string;
};

// Mapped types
type OptionalUser = {
  [P in keyof User]?: User[P];
};

// Conditional types
type NonNullableUser = {
  [P in keyof User]: User[P] extends null | undefined ? never : User[P];
};
```

## 4. Functions <a name="functions"></a>

### Function Types
```typescript
// Function declaration
function greet(name: string): string {
  return `Hello, ${name}!`;
}

// Function expression
const multiply: (a: number, b: number) => number = function(a, b) {
  return a * b;
};

// Arrow function
const divide = (a: number, b: number): number => a / b;

// Optional parameters
function buildName(firstName: string, lastName?: string): string {
  return lastName ? `${firstName} ${lastName}` : firstName;
}

// Default parameters
function createGreeting(name: string, greeting: string = "Hello"): string {
  return `${greeting}, ${name}!`;
}

// Rest parameters
function sum(...numbers: number[]): number {
  return numbers.reduce((total, num) => total + num, 0);
}

// Function overloads
function processInput(input: string): string;
function processInput(input: number): number;
function processInput(input: string | number): string | number {
  if (typeof input === "string") {
    return input.toUpperCase();
  } else {
    return input * 2;
  }
}

// This parameter
interface Card {
  suit: string;
  card: number;
}

interface Deck {
  suits: string[];
  cards: number[];
  createCardPicker(this: Deck): () => Card;
}

let deck: Deck = {
  suits: ["hearts", "spades", "clubs", "diamonds"],
  cards: Array(52),
  createCardPicker: function(this: Deck) {
    return () => {
      const pickedCard = Math.floor(Math.random() * 52);
      const pickedSuit = Math.floor(pickedCard / 13);
      
      return {
        suit: this.suits[pickedSuit],
        card: pickedCard % 13
      };
    };
  }
};
```

## 5. Classes and OOP <a name="classes-oop"></a>

### Basic Classes
```typescript
class Animal {
  // Properties
  name: string;
  private age: number;
  protected species: string;
  readonly createdAt: Date;

  // Constructor
  constructor(name: string, age: number, species: string) {
    this.name = name;
    this.age = age;
    this.species = species;
    this.createdAt = new Date();
  }

  // Methods
  speak(): string {
    return "Animal sound";
  }

  // Getter
  get ageInHumanYears(): number {
    return this.age * 7;
  }

  // Setter
  set updateAge(newAge: number) {
    if (newAge > 0) {
      this.age = newAge;
    }
  }

  // Static method
  static createUnknownAnimal(): Animal {
    return new Animal("Unknown", 0, "Unknown");
  }
}

// Inheritance
class Dog extends Animal {
  breed: string;

  constructor(name: string, age: number, breed: string) {
    super(name, age, "Canine");
    this.breed = breed;
  }

  // Method overriding
  speak(): string {
    return "Woof!";
  }

  // Accessing protected property
  getSpecies(): string {
    return this.species;
  }
}

// Abstract class
abstract class Vehicle {
  abstract start(): void;
  
  stop(): void {
    console.log("Vehicle stopped");
  }
}

class Car extends Vehicle {
  start(): void {
    console.log("Car started");
  }
}

// Interfaces with classes
interface Drivable {
  drive(): void;
}

class SportsCar implements Drivable {
  drive(): void {
    console.log("Driving fast!");
  }
}

// Using classes
const myDog = new Dog("Buddy", 3, "Golden Retriever");
console.log(myDog.speak()); // "Woof!"
console.log(myDog.ageInHumanYears); // 21

const unknownAnimal = Animal.createUnknownAnimal();
```

## 6. Generics <a name="generics"></a>

### Basic Generics
```typescript
// Generic function
function identity<T>(arg: T): T {
  return arg;
}

let output = identity<string>("hello");
let output2 = identity(42); // Type inference

// Generic interface
interface GenericIdentityFn<T> {
  (arg: T): T;
}

let myIdentity: GenericIdentityFn<number> = identity;

// Generic class
class GenericNumber<T> {
  zeroValue: T;
  add: (x: T, y: T) => T;
}

let myGenericNumber = new GenericNumber<number>();
myGenericNumber.zeroValue = 0;
myGenericNumber.add = (x, y) => x + y;

// Generic constraints
interface Lengthwise {
  length: number;
}

function loggingIdentity<T extends Lengthwise>(arg: T): T {
  console.log(arg.length);
  return arg;
}

// Using type parameters in generic constraints
function getProperty<T, K extends keyof T>(obj: T, key: K) {
  return obj[key];
}

let x = { a: 1, b: 2, c: 3 };
getProperty(x, "a"); // okay
// getProperty(x, "m"); // error

// Generic utility functions
function createArray<T>(length: number, value: T): T[] {
  return Array(length).fill(value);
}

// Generic with default type
interface ApiResponse<T = any> {
  data: T;
  status: number;
}

const response: ApiResponse<User> = {
  data: { id: 1, name: "John", email: "john@example.com", createdAt: new Date() },
  status: 200
};
```

## 7. Advanced Types <a name="advanced-types"></a>

### Type Guards
```typescript
// typeof type guard
function padLeft(value: string, padding: string | number) {
  if (typeof padding === "number") {
    return Array(padding + 1).join(" ") + value;
  }
  if (typeof padding === "string") {
    return padding + value;
  }
  throw new Error(`Expected string or number, got '${padding}'.`);
}

// instanceof type guard
class Bird {
  fly() {
    console.log("flying");
  }
}

class Fish {
  swim() {
    console.log("swimming");
  }
}

function move(pet: Bird | Fish) {
  if (pet instanceof Bird) {
    pet.fly();
  } else if (pet instanceof Fish) {
    pet.swim();
  }
}

// in operator type guard
interface Admin {
  permissions: string[];
}

function isAdmin(user: User | Admin): user is Admin {
  return (user as Admin).permissions !== undefined;
}

// User-defined type guards
function isString(test: any): test is string {
  return typeof test === "string";
}

// Nullish coalescing
let input = null;
let defaultValue = "default";
let result = input ?? defaultValue; // "default"

// Optional chaining
interface UserWithAddress {
  name: string;
  address?: {
    street: string;
    city: string;
  };
}

function getCity(user: UserWithAddress): string | undefined {
  return user.address?.city;
}

// Literal types
type Easing = "ease-in" | "ease-out" | "ease-in-out";
type NumericLiteral = 1 | 2 | 3 | 4 | 5;

// Template literal types
type World = "world";
type Greeting = `hello ${World}`; // "hello world"

type Color = "red" | "blue";
type Size = "small" | "large";
type ColoredSize = `${Color}-${Size}`; // "red-small" | "red-large" | "blue-small" | "blue-large"
```

## 8. Modules and Namespaces <a name="modules-namespaces"></a>

### ES Modules
```typescript
// math.ts
export function add(a: number, b: number): number {
  return a + b;
}

export function subtract(a: number, b: number): number {
  return a - b;
}

export const PI = 3.14159;

// Default export
export default class Calculator {
  // class implementation
}

// main.ts
import Calculator, { add, subtract, PI } from './math';
import * as math from './math';

console.log(add(2, 3));
console.log(math.PI);

// Re-exporting
export { add, subtract } from './math';

// Dynamic imports
async function loadMath() {
  const math = await import('./math');
  console.log(math.add(2, 3));
}
```

### Namespaces (legacy)
```typescript
namespace Geometry {
  export interface Point {
    x: number;
    y: number;
  }

  export function distance(p1: Point, p2: Point): number {
    return Math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2);
  }
}

// Using namespace
const point1: Geometry.Point = { x: 0, y: 0 };
const point2: Geometry.Point = { x: 3, y: 4 };
console.log(Geometry.distance(point1, point2)); // 5

// Multi-file namespaces
// file1.ts
namespace Validation {
  export interface StringValidator {
    isAcceptable(s: string): boolean;
  }
}

// file2.ts
/// <reference path="file1.ts" />
namespace Validation {
  export class LettersOnlyValidator implements StringValidator {
    isAcceptable(s: string): boolean {
      return /^[A-Za-z]+$/.test(s);
    }
  }
}
```

## 9. Decorators <a name="decorators"></a>

```typescript
// Class decorator
function sealed(constructor: Function) {
  Object.seal(constructor);
  Object.seal(constructor.prototype);
}

@sealed
class Greeter {
  greeting: string;
  
  constructor(message: string) {
    this.greeting = message;
  }
  
  greet() {
    return "Hello, " + this.greeting;
  }
}

// Method decorator
function enumerable(value: boolean) {
  return function (target: any, propertyKey: string, descriptor: PropertyDescriptor) {
    descriptor.enumerable = value;
  };
}

class Person {
  name: string;
  
  constructor(name: string) {
    this.name = name;
  }
  
  @enumerable(false)
  greet() {
    return "Hello, " + this.name;
  }
}

// Property decorator
function format(formatString: string) {
  return function (target: any, propertyKey: string): any {
    let value = target[propertyKey];
    
    function getter() {
      return `${formatString} ${value}`;
    }
    
    function setter(newVal: string) {
      value = newVal;
    }
    
    return {
      get: getter,
      set: setter,
      enumerable: true,
      configurable: true
    };
  };
}

class Product {
  @format('$')
  price: string;
  
  constructor(price: string) {
    this.price = price;
  }
}

// Parameter decorator
function validate(target: any, propertyKey: string, parameterIndex: number) {
  // Validation logic here
}

class UserService {
  createUser(@validate name: string, @validate email: string) {
    // Create user
  }
}

// Decorator factory
function log(prefix: string) {
  return (target: any) => {
    console.log(`${prefix} - ${target}`);
  };
}

@log('debug')
class MyClass {}
```

## 10. Utility Types <a name="utility-types"></a>

```typescript
interface User {
  id: number;
  name: string;
  email: string;
  age?: number;
  createdAt: Date;
}

// Partial - all properties optional
type PartialUser = Partial<User>;

// Readonly - all properties readonly
type ReadonlyUser = Readonly<User>;

// Required - all properties required
type RequiredUser = Required<User>;

// Pick - select specific properties
type UserPreview = Pick<User, 'id' | 'name'>;

// Omit - exclude specific properties
type UserWithoutId = Omit<User, 'id'>;

// Record - object with specific key type and value type
type UserMap = Record<string, User>;

// Exclude - exclude types from union
type T0 = Exclude<"a" | "b" | "c", "a">; // "b" | "c"

// Extract - extract types from union
type T1 = Extract<"a" | "b" | "c", "a" | "f">; // "a"

// NonNullable - exclude null and undefined
type T2 = NonNullable<string | number | undefined>; // string | number

// Parameters - tuple of function parameters
type T3 = Parameters<(s: string, n: number) => void>; // [string, number]

// ReturnType - return type of function
type T4 = ReturnType<() => string>; // string

// ConstructorParameters - parameters of constructor
type T5 = ConstructorParameters<typeof Error>; // [string?]

// InstanceType - instance type of constructor
type T6 = InstanceType<typeof Error>; // Error

// ThisParameterType - extract this parameter type
function toHex(this: Number) {
  return this.toString(16);
}
type T7 = ThisParameterType<typeof toHex>; // Number

// OmitThisParameter - remove this parameter
type T8 = OmitThisParameter<typeof toHex>; // () => string

// ThisType - marker for contextual this type
interface ObjectDescriptor<D, M> {
  data?: D;
  methods?: M & ThisType<D & M>;
}

function makeObject<D, M>(desc: ObjectDescriptor<D, M>): D & M {
  let data: object = desc.data || {};
  let methods: object = desc.methods || {};
  return { ...data, ...methods } as D & M;
}

let obj = makeObject({
  data: { x: 0, y: 0 },
  methods: {
    moveBy(dx: number, dy: number) {
      this.x += dx; // Strongly typed this
      this.y += dy; // Strongly typed this
    }
  }
});
```

## 11. TypeScript with React <a name="typescript-react"></a>

### Functional Components
```typescript
import React, { useState, useEffect, ReactNode } from 'react';

// Props interface
interface ButtonProps {
  children: ReactNode;
  onClick?: () => void;
  variant?: 'primary' | 'secondary';
  disabled?: boolean;
}

// Functional component
const Button: React.FC<ButtonProps> = ({
  children,
  onClick,
  variant = 'primary',
  disabled = false
}) => {
  return (
    <button
      className={`btn btn-${variant}`}
      onClick={onClick}
      disabled={disabled}
    >
      {children}
    </button>
  );
};

// Component with state
interface CounterProps {
  initialCount?: number;
}

const Counter: React.FC<CounterProps> = ({ initialCount = 0 }) => {
  const [count, setCount] = useState<number>(initialCount);
  
  const increment = () => setCount(prev => prev + 1);
  const decrement = () => setCount(prev => prev - 1);
  
  return (
    <div>
      <p>Count: {count}</p>
      <Button onClick={increment}>+</Button>
      <Button onClick={decrement}>-</Button>
    </div>
  );
};

// Component with useEffect
interface UserProfileProps {
  userId: number;
}

const UserProfile: React.FC<UserProfileProps> = ({ userId }) => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  
  useEffect(() => {
    const fetchUser = async () => {
      setLoading(true);
      try {
        const response = await fetch(`/api/users/${userId}`);
        const userData: User = await response.json();
        setUser(userData);
      } catch (error) {
        console.error('Error fetching user:', error);
      } finally {
        setLoading(false);
      }
    };
    
    fetchUser();
  }, [userId]);
  
  if (loading) return <div>Loading...</div>;
  if (!user) return <div>User not found</div>;
  
  return (
    <div>
      <h1>{user.name}</h1>
      <p>Email: {user.email}</p>
    </div>
  );
};
```

### Class Components
```typescript
import React, { Component } from 'react';

interface TimerProps {
  initialSeconds: number;
}

interface TimerState {
  seconds: number;
  isRunning: boolean;
}

class Timer extends Component<TimerProps, TimerState> {
  private intervalId?: number;
  
  constructor(props: TimerProps) {
    super(props);
    this.state = {
      seconds: props.initialSeconds,
      isRunning: false
    };
  }
  
  componentDidMount() {
    this.startTimer();
  }
  
  componentWillUnmount() {
    this.stopTimer();
  }
  
  startTimer = () => {
    this.setState({ isRunning: true });
    this.intervalId = window.setInterval(() => {
      this.setState(prevState => ({
        seconds: prevState.seconds - 1
      }));
    }, 1000);
  };
  
  stopTimer = () => {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.setState({ isRunning: false });
    }
  };
  
  render() {
    const { seconds, isRunning } = this.state;
    
    return (
      <div>
        <p>Time remaining: {seconds}s</p>
        {isRunning ? (
          <Button onClick={this.stopTimer}>Stop</Button>
        ) : (
          <Button onClick={this.startTimer}>Start</Button>
        )}
      </div>
    );
  }
}
```

### Custom Hooks
```typescript
import { useState, useEffect } from 'react';

// Custom hook with TypeScript
function useLocalStorage<T>(key: string, initialValue: T) {
  const [storedValue, setStoredValue] = useState<T>(() => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch (error) {
      console.error(error);
      return initialValue;
    }
  });
  
  const setValue = (value: T | ((val: T) => T)) => {
    try {
      const valueToStore = value instanceof Function ? value(storedValue) : value;
      setStoredValue(valueToStore);
      window.localStorage.setItem(key, JSON.stringify(valueToStore));
    } catch (error) {
      console.error(error);
    }
  };
  
  return [storedValue, setValue] as const;
}

// Usage
const [name, setName] = useLocalStorage<string>('name', 'John Doe');

// Another custom hook
function useApi<T>(url: string) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const response = await fetch(url);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const result: T = await response.json();
        setData(result);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An error occurred');
      } finally {
        setLoading(false);
      }
    };
    
    fetchData();
  }, [url]);
  
  return { data, loading, error };
}

// Usage
const { data: user, loading, error } = useApi<User>('/api/user/1');
```

## 12. Project Configuration <a name="project-configuration"></a>

### tsconfig.json
```json
{
  "compilerOptions": {
    /* Basic Options */
    "target": "ES2020",                          // Specify ECMAScript target version
    "module": "ESNext",                          // Specify module code generation
    "lib": ["DOM", "ES2020"],                    // Specify library files to be included
    "allowJs": true,                             // Allow JavaScript files to be compiled
    "checkJs": true,                             // Report errors in .js files
    "jsx": "react-jsx",                          // Specify JSX code generation
    
    /* Strict Type-Checking Options */
    "strict": true,                              // Enable all strict type-checking options
    "noImplicitAny": true,                       // Raise error on expressions and declarations with an implied 'any' type
    "strictNullChecks": true,                    // Enable strict null checks
    "strictFunctionTypes": true,                 // Enable strict checking of function types
    "strictBindCallApply": true,                 // Enable strict 'bind', 'call', and 'apply' methods on functions
    "strictPropertyInitialization": true,        // Enable strict checking of property initialization in classes
    "noImplicitThis": true,                      // Raise error on 'this' expressions with an implied 'any' type
    "alwaysStrict": true,                        // Parse in strict mode and emit "use strict" for each source file
    
    /* Additional Checks */
    "noUnusedLocals": true,                      // Report errors on unused locals
    "noUnusedParameters": true,                  // Report errors on unused parameters
    "noImplicitReturns": true,                   // Report error when not all code paths in function return a value
    "noFallthroughCasesInSwitch": true,          // Report errors for fallthrough cases in switch statement
    
    /* Module Resolution Options */
    "moduleResolution": "node",                  // Specify module resolution strategy
    "baseUrl": "./",                             // Base directory to resolve non-absolute module names
    "paths": {                                   // A series of entries which re-map imports to lookup locations
      "@/*": ["src/*"],
      "@/components/*": ["src/components/*"]
    },
    "rootDirs": ["src", "generated"],            // List of root folders whose combined content represents the structure of the project at runtime
    "typeRoots": ["node_modules/@types"],        // List of folders to include type definitions from
    "types": ["node", "jest"],                   // Type declaration files to be included in compilation
    "allowSyntheticDefaultImports": true,        // Allow default imports from modules with no default export
    "esModuleInterop": true,                     // Enables emit interoperability between CommonJS and ES Modules
    "preserveSymlinks": true,                    // Do not resolve the real path of symlinks
    
    /* Source Map Options */
    "sourceMap": true,                           // Generates corresponding '.map' file
    "inlineSourceMap": false,                    // Emit a single file with source maps instead of having a separate file
    "declaration": true,                         // Generates corresponding '.d.ts' file
    "declarationMap": true,                      // Generates a sourcemap for each corresponding '.d.ts' file
    "removeComments": false,                     // Do not emit comments to output
    
    /* Experimental Options */
    "experimentalDecorators": true,              // Enables experimental support for ES7 decorators
    "emitDecoratorMetadata": true,               // Enables experimental support for emitting type metadata for decorators
    
    /* Advanced Options */
    "skipLibCheck": true,                        // Skip type checking of declaration files
    "forceConsistentCasingInFileNames": true,    // Disallow inconsistently-cased references to the same file
    "outDir": "dist",                            // Redirect output structure to the directory
    "rootDir": "src"                             // Specify the root directory of input files
  },
  "include": [
    "src/**/*",
    "tests/**/*"
  ],
  "exclude": [
    "node_modules",
    "dist",
    "**/*.test.ts"
  ]
}
```

## 13. Best Practices <a name="best-practices"></a>

### Code Organization
```typescript
// Use interfaces for public API, type aliases for internal
export interface PublicUser {
  id: number;
  name: string;
}

type InternalUser = PublicUser & {
  passwordHash: string;
  createdAt: Date;
};

// Prefer const enums for better performance
const enum Status {
  Active = "ACTIVE",
  Inactive = "INACTIVE",
  Pending = "PENDING"
}

// Use readonly for immutable data structures
interface ImmutablePoint {
  readonly x: number;
  readonly y: number;
}

// Use branded types for additional type safety
type UserId = number & { readonly brand: unique symbol };
type Email = string & { readonly brand: unique symbol };

function createUserId(id: number): UserId {
  return id as UserId;
}

function createEmail(email: string): Email {
  if (!email.includes('@')) throw new Error('Invalid email');
  return email as Email;
}
```

### Error Handling
```typescript
// Use custom error types
class AppError extends Error {
  constructor(
    message: string,
    public code: string,
    public statusCode: number = 500
  ) {
    super(message);
    this.name = 'AppError';
  }
}

// Use Result type for better error handling
type Result<T, E = Error> = 
  | { success: true; data: T }
  | { success: false; error: E };

function safeParseJSON(json: string): Result<any, SyntaxError> {
  try {
    const data = JSON.parse(json);
    return { success: true, data };
  } catch (error) {
    return { 
      success: false, 
      error: error instanceof SyntaxError ? error : new SyntaxError('Invalid JSON') 
    };
  }
}
```

### Performance Tips
```typescript
// Use const assertions for literal types
const colors = ['red', 'green', 'blue'] as const;
type Color = typeof colors[number]; // "red" | "green" | "blue"

// Use satisfies operator for type checking without widening
const user = {
  name: "John",
  age: 30
} satisfies { name: string; age: number };

// Use template literal types for complex string patterns
type Route = `/${string}`;
type ApiRoute = `/api/${string}`;

// Use mapped types for dynamic type creation
type OptionalFields<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>;
type ReadonlyFields<T, K extends keyof T> = Omit<T, K> & Readonly<Pick<T, K>>;
```

This comprehensive guide covers TypeScript from basic concepts to advanced patterns. Remember to practice each concept and apply them in real projects to gain proficiency. TypeScript's strength lies in its type system, so leverage it to write more robust and maintainable code.
